import logging

import numpy as np
from ray import serve

logger = logging.getLogger(__name__)


@serve.deployment(
    name="LanguageDetector",
    num_replicas=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0.25},
)
class LanguageDetector:
    """Language detection using faster-whisper ``large-v3`` encoder.

    Uses only the encoder and language-token logits, so it is fast
    compared to full transcription.

    Parameters
    ----------
    model_size : str
        Hugging Face model identifier (default ``"large-v3"``).
    """

    def __init__(self, model_size: str = "large-v3") -> None:
        from faster_whisper import WhisperModel

        logger.info("Loading language detector %s (device=auto)", model_size)
        self.model = WhisperModel(
            model_size,
            device="auto",
            compute_type="auto",
            download_root="/tmp/whisper-models",
        )
        logger.info("Language detector model loaded")

    def detect(self, audio: np.ndarray) -> dict:
        """Detect the spoken language from an audio sample.

        Parameters
        ----------
        audio : np.ndarray
            Float32 samples at 16 kHz, shape ``(n_samples,)``.

        Returns
        -------
        dict
            ``{"language": str, "language_probability": float}`` on
            success, or ``{"error": str}`` on failure.
        """
        try:
            language, probability, all_probs = self.model.detect_language(audio)
            logger.info("Detected language: %s (prob=%.3f)", language, probability)
            return {
                "language": language,
                "language_probability": probability,
            }
        except Exception as e:
            logger.error("Language detection error: %s", e)
            return {"error": str(e)}
