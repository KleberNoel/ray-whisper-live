import logging

import numpy as np
from ray import serve

logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURES: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


@serve.deployment(
    name="WhisperTranscriber",
    num_replicas=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0.25},
)
class WhisperTranscriber:
    """Faster-whisper transcription on GPU using ``large-v3-turbo``.

    Parameters
    ----------
    model_size : str
        Hugging Face model identifier (e.g. ``"large-v3-turbo"``).
    """

    def __init__(self, model_size: str = "large-v3-turbo") -> None:
        from faster_whisper import WhisperModel

        logger.info("Loading Whisper %s (device=auto)", model_size)
        self.model = WhisperModel(
            model_size,
            device="auto",
            compute_type="auto",
            download_root="/tmp/whisper-models",
        )
        logger.info("Whisper model loaded")

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
        task: str = "transcribe",
        initial_prompt: str | None = None,
        beam_size: int = 5,
        no_speech_threshold: float = 0.45,
        temperature: list[float] | None = None,
        condition_on_previous_text: bool = True,
    ) -> dict:
        """Transcribe an audio chunk.

        Parameters
        ----------
        audio : np.ndarray
            Float32 samples at 16 kHz, shape ``(n_samples,)``.
        language : str or None
            BCP-47 language code.  ``None`` for auto-detection.
        task : str
            ``"transcribe"`` (default).
        initial_prompt : str or None
            Optional decoder prompt for context priming.
        beam_size : int
            Beam width (higher is more accurate but slower).
        no_speech_threshold : float
            Segments above this no-speech probability are discarded.
        temperature : list[float] or None
            Temperature fallback schedule.  Defaults to
            ``[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]``.
        condition_on_previous_text : bool
            Feed previous output as decoder context.

        Returns
        -------
        dict
            ``{"segments": [...], "language": str,
            "language_probability": float}`` on success, or
            ``{"error": str, "segments": []}`` on failure.
        """
        if temperature is None:
            temperature = DEFAULT_TEMPERATURES

        try:
            segments, info = self.model.transcribe(
                audio,
                language=language,
                task=task,
                beam_size=beam_size,
                temperature=temperature,
                condition_on_previous_text=condition_on_previous_text,
                no_speech_threshold=no_speech_threshold,
                initial_prompt=initial_prompt,
                vad_filter=False,
                word_timestamps=False,
                suppress_blank=True,
                suppress_tokens=[-1],
            )
            return {
                "segments": [
                    {
                        "start": round(s.start, 3),
                        "end": round(s.end, 3),
                        "text": s.text.strip(),
                        "no_speech_prob": s.no_speech_prob,
                    }
                    for s in segments
                ],
                "language": info.language,
                "language_probability": info.language_probability,
            }
        except Exception as e:
            logger.error("Transcription error: %s", e)
            return {"error": str(e), "segments": []}
