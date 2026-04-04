import argparse
import gc
import logging
import time

import numpy as np
import ray
from ray import serve

from src.gpu_profiler import GpuProfiler, ProfileResult
from src.language_detector import LanguageDetector
from src.server import WhisperLiveServer
from src.transcriber import WhisperTranscriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default static allocations (used in --mode static)
DEFAULT_TRANSCRIBER_GPU: float = 0.25
DEFAULT_DETECTOR_GPU: float = 0.25


def _generate_probe_audio(duration_s: float = 5.0) -> np.ndarray:
    """Generate probe audio: mixed sine tones simulating speech.

    Parameters
    ----------
    duration_s : float
        Duration in seconds.

    Returns
    -------
    np.ndarray
        Float32 audio at 16 kHz.
    """
    sr = 16000
    t = np.linspace(
        0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32
    )
    # Mix of frequencies roughly in speech range
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.3 * np.sin(2 * np.pi * 800 * t)
        + 0.2 * np.sin(2 * np.pi * 1500 * t)
        + 0.1 * np.sin(2 * np.pi * 3000 * t)
    )
    return audio


def _profile_transcriber(
    model_size: str, profiler: GpuProfiler, probe_audio: np.ndarray
) -> ProfileResult:
    """Profile GPU memory for the transcriber model.

    Loads the model, runs transcription on probe audio, captures peak
    GPU memory, then unloads.

    Parameters
    ----------
    model_size : str
        Whisper model identifier.
    profiler : GpuProfiler
        GPU memory profiler instance.
    probe_audio : np.ndarray
        Probe audio for inference.

    Returns
    -------
    ProfileResult
        Profiling result with peak memory and recommended ``num_gpus``.
    """
    from faster_whisper import WhisperModel

    logger.info("Profiling transcriber model: %s", model_size)
    profiler.start()

    model = WhisperModel(
        model_size,
        device="cuda",
        compute_type="float16",
        download_root="/tmp/whisper-models",
    )

    # Run transcription to capture inference peak
    segments, info = model.transcribe(
        probe_audio,
        language="en",
        beam_size=5,
        vad_filter=False,
    )
    # Force generator consumption
    _ = list(segments)

    # Run a second pass to capture stable peak
    segments, info = model.transcribe(
        probe_audio,
        language="en",
        beam_size=5,
        vad_filter=False,
    )
    _ = list(segments)

    result = profiler.stop()
    logger.info(
        "Transcriber %s: delta=%.2f GiB, num_gpus=%.2f",
        model_size,
        result.delta_bytes / (1024**3),
        result.num_gpus,
    )

    # Unload model
    del model
    gc.collect()

    # Give GPU time to release memory
    try:
        import ctranslate2

        ctranslate2.empty_cuda_cache()
    except (ImportError, AttributeError):
        pass

    time.sleep(1.0)
    return result


def _profile_detector(
    model_size: str, profiler: GpuProfiler, probe_audio: np.ndarray
) -> ProfileResult:
    """Profile GPU memory for the language detector model.

    Loads the model, runs language detection on probe audio, captures peak
    GPU memory, then unloads.

    Parameters
    ----------
    model_size : str
        Whisper model identifier.
    profiler : GpuProfiler
        GPU memory profiler instance.
    probe_audio : np.ndarray
        Probe audio for detection.

    Returns
    -------
    ProfileResult
        Profiling result with peak memory and recommended ``num_gpus``.
    """
    from faster_whisper import WhisperModel

    logger.info("Profiling language detector model: %s", model_size)
    profiler.start()

    model = WhisperModel(
        model_size,
        device="cuda",
        compute_type="float16",
        download_root="/tmp/whisper-models",
    )

    # Run detection to capture inference peak
    _ = model.detect_language(probe_audio)
    # Second pass for stable peak
    _ = model.detect_language(probe_audio)

    result = profiler.stop()
    logger.info(
        "Detector %s: delta=%.2f GiB, num_gpus=%.2f",
        model_size,
        result.delta_bytes / (1024**3),
        result.num_gpus,
    )

    # Unload model
    del model
    gc.collect()

    try:
        import ctranslate2

        ctranslate2.empty_cuda_cache()
    except (ImportError, AttributeError):
        pass

    time.sleep(1.0)
    return result


def _run_dynamic_profiling(
    model_size: str, lang_model_size: str
) -> tuple[float, float]:
    """Profile both models and return recommended num_gpus fractions.

    Parameters
    ----------
    model_size : str
        Transcriber model identifier.
    lang_model_size : str
        Language detector model identifier.

    Returns
    -------
    tuple[float, float]
        ``(transcriber_gpu, detector_gpu)`` fractions for Ray.
    """
    profiler = GpuProfiler(poll_interval_ms=5.0, margin=1.15)
    probe_audio = _generate_probe_audio(duration_s=5.0)

    logger.info(
        "=== Dynamic GPU profiling (total=%.2f GiB) ===",
        profiler.total_bytes / (1024**3),
    )

    # Profile transcriber first
    transcriber_result = _profile_transcriber(model_size, profiler, probe_audio)

    # Profile detector (may share weights if same model family)
    detector_result = _profile_detector(lang_model_size, profiler, probe_audio)

    # Validate: both models need to fit on the GPU simultaneously
    total_fraction = transcriber_result.num_gpus + detector_result.num_gpus
    if total_fraction > 1.0:
        logger.warning(
            "Combined GPU fraction %.2f exceeds 1.0! "
            "Models may not fit simultaneously. "
            "Scaling down proportionally.",
            total_fraction,
        )
        scale = 0.95 / total_fraction
        transcriber_gpu = round(transcriber_result.num_gpus * scale * 20) / 20
        detector_gpu = round(detector_result.num_gpus * scale * 20) / 20
    else:
        transcriber_gpu = transcriber_result.num_gpus
        detector_gpu = detector_result.num_gpus

    logger.info(
        "=== Dynamic profiling complete ===\n"
        "  Transcriber (%s): %.2f GiB -> num_gpus=%.2f\n"
        "  Detector    (%s): %.2f GiB -> num_gpus=%.2f\n"
        "  Total GPU fraction: %.2f",
        model_size,
        transcriber_result.delta_bytes / (1024**3),
        transcriber_gpu,
        lang_model_size,
        detector_result.delta_bytes / (1024**3),
        detector_gpu,
        transcriber_gpu + detector_gpu,
    )

    profiler.shutdown()
    return transcriber_gpu, detector_gpu


def main() -> None:
    """Start the Ray Serve WhisperLive application with 3 deployments."""
    parser = argparse.ArgumentParser(description="Ray Serve WhisperLive")
    parser.add_argument(
        "--model-size", default="large-v3-turbo", help="Whisper transcription model"
    )
    parser.add_argument(
        "--lang-model-size", default="large-v3", help="Whisper language detection model"
    )
    parser.add_argument(
        "--mode",
        choices=["static", "dynamic"],
        default="static",
        help=(
            "GPU allocation mode: 'static' uses fixed num_gpus=0.25, "
            "'dynamic' profiles actual GPU memory usage before deploying"
        ),
    )
    args = parser.parse_args()

    # Determine GPU fractions
    if args.mode == "dynamic":
        logger.info("Mode: dynamic — profiling GPU memory before deployment")
        transcriber_gpu, detector_gpu = _run_dynamic_profiling(
            args.model_size, args.lang_model_size
        )
    else:
        logger.info("Mode: static — using fixed GPU allocations")
        transcriber_gpu = DEFAULT_TRANSCRIBER_GPU
        detector_gpu = DEFAULT_DETECTOR_GPU

    ray.init(num_cpus=4, num_gpus=1, ignore_reinit_error=True)
    logger.info(
        "Ray: %.0f GPUs, %.0f CPUs",
        ray.available_resources().get("GPU", 0),
        ray.available_resources().get("CPU", 0),
    )

    # Apply dynamic GPU fractions via .options()
    transcriber = WhisperTranscriber.options(
        ray_actor_options={"num_cpus": 1, "num_gpus": transcriber_gpu},
    ).bind(model_size=args.model_size)

    language_detector = LanguageDetector.options(
        ray_actor_options={"num_cpus": 1, "num_gpus": detector_gpu},
    ).bind(model_size=args.lang_model_size)

    server = WhisperLiveServer.bind(
        transcriber_handle=transcriber,
        language_detector_handle=language_detector,
    )
    serve.run(server, route_prefix="/")

    logger.info(
        "WhisperLive ready: ws://localhost:8000/listen "
        "(transcriber=%s [%.2f GPU], lang_detector=%s [%.2f GPU], mode=%s)",
        args.model_size,
        transcriber_gpu,
        args.lang_model_size,
        detector_gpu,
        args.mode,
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
