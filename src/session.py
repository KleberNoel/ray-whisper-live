import logging

import numpy as np
from fastapi import WebSocket
from faster_whisper.vad import VadOptions

from src.config import (
    DISCARD_SECONDS,
    MAX_BUFFER_SECONDS,
    SAMPLE_RATE,
    AsrConfig,
)

logger = logging.getLogger(__name__)


class ClientSession:
    """Manages audio state and configuration for a single WebSocket client.

    Parameters
    ----------
    uid : str
        Unique session identifier.
    websocket : WebSocket
        The client's WebSocket connection.
    language : str or None
        BCP-47 language code, or ``None`` to let Whisper auto-detect.
    task : str
        ``"transcribe"`` (default).
    initial_prompt : str or None
        Optional decoder prompt for context priming.
    use_vad : bool
        Gate transcription on voice activity.
    vad : VadOptions or None
        Silero VAD configuration.
    asr : AsrConfig or None
        ASR decode settings.
    """

    def __init__(
        self,
        uid: str,
        websocket: WebSocket,
        *,
        language: str | None = None,
        task: str = "transcribe",
        initial_prompt: str | None = None,
        use_vad: bool = True,
        vad: VadOptions | None = None,
        asr: AsrConfig | None = None,
    ) -> None:
        self.uid = uid
        self.websocket = websocket
        self.language = language
        self.task = task
        self.initial_prompt = initial_prompt
        self.use_vad = use_vad
        self.vad = vad or VadOptions()
        self.asr = asr or AsrConfig()

        # Audio buffer
        self.audio_buffer: np.ndarray = np.array([], dtype=np.float32)
        self.timestamp_offset: float = 0.0
        self.frames_offset: float = 0.0

        # Partial-segment tracking
        self.current_out: str = ""
        self.prev_out: str = ""
        self.same_output_count: int = 0

        # VAD state
        self.no_voice_activity_chunks: int = 0

        # Connection state
        self.connected: bool = True
        self.last_completed_end: float = -1.0

    def add_frames(self, frame: np.ndarray) -> None:
        """Append audio frames, trimming when the buffer exceeds *MAX_BUFFER_SECONDS*.

        Parameters
        ----------
        frame : np.ndarray
            Float32 audio samples at 16 kHz.
        """
        if self.audio_buffer.size > 0:
            self.audio_buffer = np.concatenate([self.audio_buffer, frame], axis=0)
        else:
            self.audio_buffer = frame.copy()

        max_samples = MAX_BUFFER_SECONDS * SAMPLE_RATE
        if self.audio_buffer.shape[0] > max_samples:
            self.audio_buffer = self.audio_buffer[int(DISCARD_SECONDS * SAMPLE_RATE) :]
            self.frames_offset += DISCARD_SECONDS
            if self.timestamp_offset < self.frames_offset:
                self.timestamp_offset = self.frames_offset

    def get_audio_chunk(self) -> tuple[np.ndarray | None, float]:
        """Return the un-transcribed tail of the buffer.

        Returns
        -------
        tuple[np.ndarray | None, float]
            ``(chunk, duration_seconds)`` or ``(None, 0.0)`` if nothing new.
        """
        samples_taken = max(
            0, int((self.timestamp_offset - self.frames_offset) * SAMPLE_RATE)
        )
        if samples_taken >= self.audio_buffer.shape[0]:
            return None, 0.0
        chunk = self.audio_buffer[samples_taken:].copy()
        return chunk, chunk.shape[0] / SAMPLE_RATE

    async def send_response(self, segments: list[dict]) -> None:
        """Send transcription segments to the client.

        Parameters
        ----------
        segments : list[dict]
            Segment dicts with ``start``, ``end``, ``text``, ``completed``.
        """
        if not self.connected:
            return
        try:
            await self.websocket.send_json({"uid": self.uid, "segments": segments})
        except Exception as e:
            logger.error("Send error for %s: %s", self.uid, e)
            self.connected = False
