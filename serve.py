import argparse
import gc
import logging
import time

import numpy as np
import ray
from ray import serve

from src.gpu_profiler import GpuProfiler, ProfileResult
from src.server import WhisperLiveServer
from src.transcriber import WhisperTranscriber
from src.vad import SileroVadDeployment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default static allocation (used in --mode static)
DEFAULT_TRANSCRIBER_GPU: float = 0.25


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


def _run_dynamic_profiling(model_size: str) -> float:
    """Profile the transcriber model and return recommended num_gpus fraction.

    Parameters
    ----------
    model_size : str
        Transcriber model identifier.

    Returns
    -------
    float
        Recommended ``num_gpus`` fraction for Ray.
    """
    profiler = GpuProfiler(poll_interval_ms=5.0, margin=1.15)
    probe_audio = _generate_probe_audio(duration_s=5.0)

    logger.info(
        "=== Dynamic GPU profiling (total=%.2f GiB) ===",
        profiler.total_bytes / (1024**3),
    )

    transcriber_result = _profile_transcriber(model_size, profiler, probe_audio)

    logger.info(
        "=== Dynamic profiling complete ===\n"
        "  Transcriber (%s): %.2f GiB -> num_gpus=%.2f",
        model_size,
        transcriber_result.delta_bytes / (1024**3),
        transcriber_result.num_gpus,
    )

    profiler.shutdown()
    return transcriber_result.num_gpus


def main() -> None:
    """Start the Ray Serve WhisperLive application with 3 deployments."""
    parser = argparse.ArgumentParser(description="Ray Serve WhisperLive")
    parser.add_argument(
        "--model-size", default="large-v3-turbo", help="Whisper transcription model"
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

    # Determine GPU fraction for the transcriber
    if args.mode == "dynamic":
        logger.info("Mode: dynamic — profiling GPU memory before deployment")
        transcriber_gpu = _run_dynamic_profiling(args.model_size)
    else:
        logger.info("Mode: static — using fixed GPU allocations")
        transcriber_gpu = DEFAULT_TRANSCRIBER_GPU

    ray.init(num_cpus=4, num_gpus=1, ignore_reinit_error=True)
    logger.info(
        "Ray: %.0f GPUs, %.0f CPUs",
        ray.available_resources().get("GPU", 0),
        ray.available_resources().get("CPU", 0),
    )

    # Transcriber — GPU
    transcriber = WhisperTranscriber.options(
        ray_actor_options={"num_cpus": 1, "num_gpus": transcriber_gpu},
    ).bind(model_size=args.model_size)

    # Silero VAD — CPU only (ONNX)
    vad = SileroVadDeployment.bind()

    server = WhisperLiveServer.bind(
        transcriber_handle=transcriber,
        vad_handle=vad,
    )
    serve.run(server, route_prefix="/")

    logger.info(
        "WhisperLive ready: ws://localhost:8000/listen "
        "(transcriber=%s [%.2f GPU], vad=SileroVad [CPU], mode=%s)",
        args.model_size,
        transcriber_gpu,
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
