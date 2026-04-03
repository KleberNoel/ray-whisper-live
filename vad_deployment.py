"""VAD (Voice Activity Detection) deployment using Silero VAD ONNX model."""

import logging
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime

from ray import serve

logger = logging.getLogger(__name__)


class SileroVAD:
    """Silero VAD model wrapper for stateless voice activity detection."""

    def __init__(self, model_path: str):
        opts = onnxruntime.SessionOptions()
        opts.log_severity_level = 3
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self.sample_rate = 16000
        self.frame_size = 512
        self.context_size = 64

    def __call__(self, audio_chunk: np.ndarray) -> float:
        """Run VAD over a chunk and return the max speech probability."""
        audio = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return 0.0

        if audio.size % self.frame_size:
            pad = self.frame_size - (audio.size % self.frame_size)
            audio = np.pad(audio, (0, pad))

        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, self.context_size), dtype=np.float32)
        audio = audio[np.newaxis, :]
        max_prob = 0.0

        for offset in range(0, audio.shape[1], self.frame_size):
            frame = audio[:, offset : offset + self.frame_size]
            x = np.concatenate([context, frame], axis=1)
            ort_inputs = {
                "input": x,
                "state": state,
                "sr": np.array(self.sample_rate, dtype=np.int64),
            }
            out, state = self.session.run(None, ort_inputs)
            context = x[:, -self.context_size :]
            max_prob = max(max_prob, float(out[0, 0]))

        return max_prob


@serve.deployment(
    name="VADDetector",
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5, "num_gpus": 0},
)
class VADDetector:
    """Ray Serve deployment for Silero VAD voice activity detection."""

    VAD_MODEL_URL = (
        "https://github.com/snakers4/silero-vad/raw/v5.0/files/silero_vad.onnx"
    )

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.vad = None
        self._load_model()

    def _load_model(self):
        """Download and load the Silero VAD ONNX model."""
        cache_dir = Path.home() / ".cache" / "ray-whisper-live"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "silero_vad.onnx"

        if not model_path.exists():
            logger.info("Downloading Silero VAD model...")
            subprocess.run(
                ["wget", "-O", str(model_path), self.VAD_MODEL_URL],
                check=True,
                capture_output=True,
            )
            logger.info(f"VAD model saved to {model_path}")

        self.vad = SileroVAD(str(model_path))
        logger.info("Silero VAD model loaded successfully")

    def detect(self, audio_frame: np.ndarray) -> dict:
        """
        Detect voice activity in an audio frame.

        Args:
            audio_frame: Audio samples at 16kHz, shape (512,)

        Returns:
            Dict with speech probability and voice activity boolean
        """
        if self.vad is None:
            return {"speech_prob": 0.0, "is_speech": False}

        prob = self.vad(audio_frame)
        return {
            "speech_prob": prob,
            "is_speech": prob > self.threshold,
        }

    def health_check(self) -> dict:
        """Health check endpoint."""
        return {"status": "healthy", "model": "silero_vad", "threshold": self.threshold}
