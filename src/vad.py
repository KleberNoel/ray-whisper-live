import logging

import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps, get_vad_model
from ray import serve

logger = logging.getLogger(__name__)


@serve.deployment(
    name="SileroVad",
    num_replicas=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0},
)
class SileroVadDeployment:
    """Silero VAD as a standalone Ray Serve deployment (CPU, ONNX).

    Wraps ``faster_whisper.vad`` so that voice-activity detection is
    a dedicated, independently scalable service rather than inline logic
    inside the WebSocket server.

    The ONNX model is loaded once at construction time and reused for
    every call.
    """

    def __init__(self) -> None:
        logger.info("Loading Silero VAD model (ONNX, CPU)")
        self._model = get_vad_model()
        logger.info("Silero VAD model loaded")

    def has_speech(
        self,
        audio: np.ndarray,
        *,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 2000,
        speech_pad_ms: int = 400,
    ) -> bool:
        """Check whether *audio* contains speech.

        Intended for short tails (e.g. last 2048 samples) to gate whether
        the server should proceed with transcription.

        Parameters
        ----------
        audio : np.ndarray
            Float32 samples at 16 kHz.
        threshold : float
            VAD probability threshold (default ``0.5``).
        min_silence_duration_ms : int
            Minimum silence duration in ms (default ``2000``).
        speech_pad_ms : int
            Padding around speech segments in ms (default ``400``).

        Returns
        -------
        bool
            ``True`` if speech is detected or the audio is too short
            to evaluate (< 512 samples).
        """
        if audio.shape[0] < 512:
            return True

        opts = VadOptions(
            threshold=threshold,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        timestamps = get_speech_timestamps(audio, opts)
        return len(timestamps) > 0

    def get_speech_segments(
        self,
        audio: np.ndarray,
        *,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 2000,
        speech_pad_ms: int = 400,
    ) -> list[dict]:
        """Return speech segment boundaries in *audio*.

        Parameters
        ----------
        audio : np.ndarray
            Float32 samples at 16 kHz.
        threshold : float
            VAD probability threshold (default ``0.5``).
        min_silence_duration_ms : int
            Minimum silence duration in ms (default ``2000``).
        speech_pad_ms : int
            Padding around speech segments in ms (default ``400``).

        Returns
        -------
        list[dict]
            Each dict has ``{"start": int, "end": int}`` sample indices.
        """
        opts = VadOptions(
            threshold=threshold,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        return get_speech_timestamps(audio, opts)
