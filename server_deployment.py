import asyncio
import logging
from dataclasses import dataclass, field

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper.vad import VadOptions, get_speech_timestamps
from ray import serve

logger = logging.getLogger(__name__)

SAMPLE_RATE: int = 16000
MIN_AUDIO_DURATION: float = 1.0
MAX_BUFFER_SECONDS: int = 45
DISCARD_SECONDS: int = 30
SAME_OUTPUT_THRESHOLD: int = 10


@dataclass
class AsrConfig:
    """ASR settings forwarded to :class:`WhisperTranscriber`.

    Parameters
    ----------
    beam_size : int
        Beam width for decoding.
    no_speech_threshold : float
        Segments with no-speech probability above this are discarded.
    temperature : list[float]
        Temperature fallback schedule for decoding.
    condition_on_previous_text : bool
        Feed previous output as decoder context.
    """

    beam_size: int = 5
    no_speech_threshold: float = 0.45
    temperature: list[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    condition_on_previous_text: bool = True


class ClientSession:
    """Manages audio state and configuration for a single WebSocket client."""

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
        """Append audio frames, trimming when the buffer exceeds *MAX_BUFFER_SECONDS*."""
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
        """Return the un-transcribed tail of the buffer, or ``(None, 0.0)``."""
        samples_taken = max(
            0, int((self.timestamp_offset - self.frames_offset) * SAMPLE_RATE)
        )
        if samples_taken >= self.audio_buffer.shape[0]:
            return None, 0.0
        chunk = self.audio_buffer[samples_taken:].copy()
        return chunk, chunk.shape[0] / SAMPLE_RATE

    async def send_response(self, segments: list[dict]) -> None:
        """Send transcription segments to the client."""
        if not self.connected:
            return
        try:
            await self.websocket.send_json({"uid": self.uid, "segments": segments})
        except Exception as e:
            logger.error("Send error for %s: %s", self.uid, e)
            self.connected = False


app = FastAPI()


@serve.deployment(
    name="WhisperLiveServer",
    num_replicas=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0},
)
@serve.ingress(app)
class WhisperLiveServer:
    """WebSocket ingress that orchestrates Silero VAD and Whisper ASR.

    VAD runs inline (CPU, via ``faster_whisper.vad``).  Transcription is
    delegated to :class:`WhisperTranscriber` over Ray.
    """

    def __init__(self, transcriber_handle) -> None:  # noqa: ANN001
        self.transcriber_handle = transcriber_handle
        self.sessions: dict[str, ClientSession] = {}

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health(self) -> dict:
        """Liveness probe."""
        return {"status": "ok", "service": "whisper-live"}

    @app.websocket("/listen")
    async def listen(self, websocket: WebSocket) -> None:
        """Accept a WebSocket, negotiate options, and stream transcriptions."""
        await websocket.accept()
        uid: str | None = None
        session: ClientSession | None = None

        try:
            options: dict = await websocket.receive_json()
            uid = str(options.get("uid", "unknown"))

            # Build per-session VAD / ASR config from client options
            vad_cfg = VadOptions(
                threshold=float(options.get("vad_threshold", 0.5)),
                min_silence_duration_ms=int(
                    options.get("min_silence_duration_ms", 2000)
                ),
                speech_pad_ms=int(options.get("speech_pad_ms", 400)),
            )

            temp_raw = options.get("temperature")
            if temp_raw is None:
                temp_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            elif isinstance(temp_raw, (int, float)):
                temp_list = [float(temp_raw)]
            else:
                temp_list = [float(t) for t in temp_raw]

            asr_cfg = AsrConfig(
                beam_size=int(options.get("beam_size", 5)),
                no_speech_threshold=float(options.get("no_speech_threshold", 0.45)),
                temperature=temp_list,
                condition_on_previous_text=bool(
                    options.get("condition_on_previous_text", True)
                ),
            )

            session = ClientSession(
                uid,
                websocket,
                language=options.get("language"),
                task=options.get("task", "transcribe"),
                initial_prompt=options.get("initial_prompt"),
                use_vad=bool(options.get("use_vad", True)),
                vad=vad_cfg,
                asr=asr_cfg,
            )

            self.sessions[uid] = session
            logger.info(
                "Client %s connected (vad=%s, beam=%d, no_speech=%.2f)",
                uid,
                session.use_vad,
                asr_cfg.beam_size,
                asr_cfg.no_speech_threshold,
            )

            await websocket.send_json(
                {"uid": uid, "message": "SERVER_READY", "backend": "faster_whisper"}
            )
            await self._audio_loop(session)

        except WebSocketDisconnect:
            logger.info("Client %s disconnected", uid)
        except Exception as exc:
            logger.exception("Error handling client %s", uid)
            if session and session.connected:
                try:
                    await session.websocket.send_json(
                        {"uid": session.uid, "status": "ERROR", "message": str(exc)}
                    )
                except Exception:
                    pass
        finally:
            try:
                if websocket.client_state.name != "DISCONNECTED":
                    await websocket.close()
            except Exception:
                pass
            if uid and uid in self.sessions:
                del self.sessions[uid]
            logger.info("Client %s cleaned up", uid)


    async def _audio_loop(self, session: ClientSession) -> None:
        """Receive audio frames and transcribe when ready."""
        while session.connected:
            try:
                data: bytes = await session.websocket.receive_bytes()
            except WebSocketDisconnect:
                session.connected = False
                break
            except Exception:
                break

            if data == b"END_OF_AUDIO":
                await self._process_remaining(session)
                break

            frame = np.frombuffer(data, dtype=np.float32)
            session.add_frames(frame)

            if session.use_vad and not self._has_speech(session):
                continue

            await self._transcribe_if_ready(session)

    def _has_speech(self, session: ClientSession) -> bool:
        """Check the buffer tail for speech using Silero VAD."""
        if session.audio_buffer.shape[0] < 512:
            return True
        tail = session.audio_buffer[-2048:]
        timestamps = get_speech_timestamps(tail, session.vad)
        if not timestamps:
            session.no_voice_activity_chunks += 1
            return False
        session.no_voice_activity_chunks = 0
        return True

    async def _call_transcriber(
        self, session: ClientSession, audio: np.ndarray
    ) -> dict:
        """Dispatch a transcription request to the Ray transcriber."""
        return await self.transcriber_handle.transcribe.remote(
            audio=audio,
            language=session.language,
            task=session.task,
            initial_prompt=session.initial_prompt,
            beam_size=session.asr.beam_size,
            no_speech_threshold=session.asr.no_speech_threshold,
            temperature=session.asr.temperature,
            condition_on_previous_text=session.asr.condition_on_previous_text,
        )

    async def _transcribe_if_ready(self, session: ClientSession) -> None:
        """Transcribe when at least *MIN_AUDIO_DURATION* seconds are buffered."""
        chunk, duration = session.get_audio_chunk()
        if chunk is None or duration < MIN_AUDIO_DURATION:
            return

        result = await self._call_transcriber(session, chunk)

        if "error" in result:
            logger.error("Transcription error for %s: %s", session.uid, result["error"])
            return

        segments: list[dict] = result.get("segments", [])
        if not segments:
            session.timestamp_offset += duration
            return

        # Auto-detect language on first successful result
        if session.language is None and result.get("language"):
            session.language = result["language"]
            await session.websocket.send_json(
                {
                    "uid": session.uid,
                    "language": result["language"],
                    "language_prob": result.get("language_probability", 0.0),
                }
            )

        no_speech_thresh = session.asr.no_speech_threshold
        response_segments: list[dict] = []
        last_completed_offset: float = 0.0

        # Completed segments (all but the last)
        for seg in segments[:-1]:
            if seg.get("no_speech_prob", 1.0) > no_speech_thresh:
                continue
            start = session.timestamp_offset + seg["start"]
            end = session.timestamp_offset + seg["end"]
            if end <= session.last_completed_end:
                continue
            response_segments.append(
                {
                    "start": f"{start:.3f}",
                    "end": f"{end:.3f}",
                    "text": seg["text"],
                    "completed": True,
                }
            )
            session.last_completed_end = end
            last_completed_offset = seg["end"]

        if last_completed_offset > 0:
            session.timestamp_offset += last_completed_offset

        # Partial segment (the last one)
        last_seg = segments[-1]
        if last_seg.get("no_speech_prob", 1.0) <= no_speech_thresh:
            session.current_out = last_seg["text"]
            start = session.timestamp_offset + last_seg["start"]
            end = session.timestamp_offset + last_seg["end"]
            response_segments.append(
                {
                    "start": f"{start:.3f}",
                    "end": f"{end:.3f}",
                    "text": session.current_out,
                    "completed": False,
                }
            )

            if session.current_out.strip() == session.prev_out.strip():
                session.same_output_count += 1
            else:
                session.same_output_count = 0

            if session.same_output_count > SAME_OUTPUT_THRESHOLD:
                session.last_completed_end = end
                session.timestamp_offset += last_seg["end"]
                session.current_out = ""
                session.same_output_count = 0
            else:
                session.prev_out = session.current_out
        else:
            session.timestamp_offset += duration

        if response_segments:
            await session.send_response(response_segments)

    async def _process_remaining(self, session: ClientSession) -> None:
        """Flush any un-transcribed audio left in the buffer."""
        chunk, duration = session.get_audio_chunk()
        if chunk is None or duration <= 0:
            return

        result = await self._call_transcriber(session, chunk)

        segments: list[dict] = result.get("segments", [])
        response_segments: list[dict] = []
        for seg in segments:
            start = session.timestamp_offset + seg["start"]
            end = session.timestamp_offset + seg["end"]
            response_segments.append(
                {
                    "start": f"{start:.3f}",
                    "end": f"{end:.3f}",
                    "text": seg["text"],
                    "completed": True,
                }
            )

        if response_segments:
            await session.send_response(response_segments)
