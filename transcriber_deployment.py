"""Whisper transcription deployment using faster-whisper."""

import logging
import shutil
import threading
import ctypes
from typing import Optional

import numpy as np

from ray import serve

logger = logging.getLogger(__name__)


def _cuda_available() -> bool:
    """Check if CUDA is actually usable by CTranslate2."""
    if shutil.which("nvidia-smi") is None:
        return False

    required_libs = ["libcuda.so.1", "libcublas.so.12"]
    for lib in required_libs:
        try:
            ctypes.CDLL(lib)
        except OSError:
            return False

    return True


class WhisperTranscriber:
    """faster-whisper transcription engine."""

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        from faster_whisper import WhisperModel

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

        logger.info(f"Loading Whisper model: {model_size} on {device} ({compute_type})")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root="/tmp/whisper-models",
        )
        logger.info("Whisper model loaded successfully")

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        no_speech_threshold: float = 0.45,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[str] = None,
    ) -> dict:
        """
        Transcribe audio with high-accuracy settings.

        Args:
            audio: Audio samples at 16kHz
            language: Language code (None for auto-detect)
            task: "transcribe" or "translate"
            beam_size: Beam size for decoding (5 for high accuracy)
            no_speech_threshold: Threshold for filtering silent segments
            condition_on_previous_text: Use previous text as context
            initial_prompt: Optional prompt to guide transcription

        Returns:
            Dict with segments, language info, and metadata
        """
        temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        segments, info = self.model.transcribe(
            audio,
            language=language,
            task=task,
            beam_size=beam_size,
            temperature=temperatures,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt,
            vad_filter=False,
            word_timestamps=False,
            suppress_blank=True,
            suppress_tokens=[-1],
        )

        result_segments = []
        for seg in segments:
            result_segments.append(
                {
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text.strip(),
                    "no_speech_prob": seg.no_speech_prob,
                    "avg_logprob": seg.avg_logprob,
                }
            )

        return {
            "segments": result_segments,
            "language": info.language if info else None,
            "language_probability": info.language_probability if info else None,
            "duration": info.duration if info else None,
        }


@serve.deployment(
    name="WhisperTranscriber",
    num_replicas=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0.25},
)
class WhisperTranscriberDeployment:
    """Ray Serve deployment for Whisper transcription."""

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        beam_size: int = 5,
        no_speech_threshold: float = 0.45,
    ):
        self.model_size = model_size
        self.beam_size = beam_size
        self.no_speech_threshold = no_speech_threshold
        self.engine = None
        self._lock = threading.Lock()
        self._load_model()

    def _load_model(self):
        """Load the Whisper model."""
        device = "cuda" if _cuda_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        self.engine = WhisperTranscriber(
            model_size=self.model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info(
            f"WhisperTranscriberDeployment ready (device={device}, "
            f"compute_type={compute_type})"
        )

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
    ) -> dict:
        """
        Transcribe audio chunk.

        Args:
            audio: Audio samples at 16kHz
            language: Language code (None for auto-detect)
            task: "transcribe" or "translate"
            initial_prompt: Optional prompt to guide transcription

        Returns:
            Dict with transcription segments and metadata
        """
        if self.engine is None:
            return {"error": "Model not loaded", "segments": []}

        with self._lock:
            try:
                result = self.engine.transcribe(
                    audio=audio,
                    language=language,
                    task=task,
                    beam_size=self.beam_size,
                    no_speech_threshold=self.no_speech_threshold,
                    condition_on_previous_text=True,
                    initial_prompt=initial_prompt,
                )
                return result
            except Exception as e:
                logger.error(f"Transcription error: {e}")
                return {"error": str(e), "segments": []}

    def health_check(self) -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model": self.model_size,
            "beam_size": self.beam_size,
        }
