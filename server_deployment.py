"""WhisperLive server deployment - WebSocket orchestrator."""

import asyncio
import logging
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from ray import serve

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
VAD_FRAME_SIZE = 512
MIN_AUDIO_DURATION = 1.0
MAX_BUFFER_SECONDS = 45
DISCARD_SECONDS = 30


class ClientSession:
    """Manages state for a single WebSocket client."""

    def __init__(self, uid: str, websocket: WebSocket):
        self.uid = uid
        self.websocket = websocket
        self.audio_buffer = np.array([], dtype=np.float32)
        self.timestamp_offset = 0.0
        self.frames_offset = 0.0
        self.language: Optional[str] = None
        self.task: str = "transcribe"
        self.initial_prompt: Optional[str] = None
        self.transcript = []
        self.current_out = ""
        self.prev_out = ""
        self.same_output_count = 0
        self.same_output_threshold = 10
        self.no_speech_thresh = 0.45
        self.use_vad = True
        self.no_voice_activity_chunks = 0
        self.eos = False
        self.connected = True
        self.last_completed_end = -1.0

    def add_frames(self, frame: np.ndarray):
        """Add audio frames to the buffer."""
        if self.audio_buffer.size > 0:
            self.audio_buffer = np.concatenate([self.audio_buffer, frame], axis=0)
        else:
            self.audio_buffer = frame.copy()

        self._trim_buffer()

    def _trim_buffer(self):
        """Trim buffer if it exceeds max duration."""
        max_samples = MAX_BUFFER_SECONDS * SAMPLE_RATE
        if self.audio_buffer.shape[0] > max_samples:
            discard_samples = int(DISCARD_SECONDS * SAMPLE_RATE)
            self.audio_buffer = self.audio_buffer[discard_samples:]
            self.frames_offset += DISCARD_SECONDS
            if self.timestamp_offset < self.frames_offset:
                self.timestamp_offset = self.frames_offset

    def get_audio_chunk(self):
        """Get unprocessed audio chunk for transcription."""
        samples_taken = max(
            0, int((self.timestamp_offset - self.frames_offset) * SAMPLE_RATE)
        )
        if samples_taken >= self.audio_buffer.shape[0]:
            return None, 0.0

        chunk = self.audio_buffer[samples_taken:].copy()
        duration = chunk.shape[0] / SAMPLE_RATE
        return chunk, duration

    def update_offset(self, offset: float):
        """Update timestamp offset after processing."""
        self.timestamp_offset += offset

    def format_segment(
        self, start: float, end: float, text: str, completed: bool = False
    ) -> dict:
        """Format a transcription segment."""
        return {
            "start": f"{start:.3f}",
            "end": f"{end:.3f}",
            "text": text,
            "completed": completed,
        }

    async def send_response(self, segments: list):
        """Send transcription result to client."""
        if not self.connected:
            return
        try:
            await self.websocket.send_json(
                {
                    "uid": self.uid,
                    "segments": segments,
                }
            )
        except Exception as e:
            logger.error(f"Error sending to client {self.uid}: {e}")
            self.connected = False

    async def send_status(self, status: str, message: str):
        """Send status message to client."""
        if not self.connected:
            return
        try:
            await self.websocket.send_json(
                {
                    "uid": self.uid,
                    "status": status,
                    "message": message,
                }
            )
        except Exception:
            self.connected = False


app = FastAPI()


@serve.deployment(
    name="WhisperLiveServer",
    num_replicas=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0},
)
@serve.ingress(app)
class WhisperLiveServer:
    """Ray Serve deployment for WhisperLive WebSocket server."""

    def __init__(self, vad_handle, transcriber_handle):
        self.vad_handle = vad_handle
        self.transcriber_handle = transcriber_handle
        self.sessions: dict[str, ClientSession] = {}

    @app.get("/health")
    async def health(self):
        return {"status": "ok", "service": "whisper-live"}

    @app.websocket("/listen")
    async def listen(self, websocket: WebSocket):
        await websocket.accept()
        uid = None
        session = None

        try:
            options = await websocket.receive_json()
            uid = options.get("uid", "unknown")
            session = ClientSession(uid, websocket)
            session.language = options.get("language")
            session.task = options.get("task", "transcribe")
            session.initial_prompt = options.get("initial_prompt")
            session.use_vad = options.get("use_vad", True)
            session.no_speech_thresh = options.get("no_speech_thresh", 0.45)
            session.same_output_threshold = options.get("same_output_threshold", 10)

            self.sessions[uid] = session
            logger.info("Client %s connected", uid)

            await websocket.send_json(
                {
                    "uid": uid,
                    "message": "SERVER_READY",
                    "backend": "faster_whisper",
                }
            )

            await self._audio_loop(session)

        except WebSocketDisconnect:
            logger.info("Client %s disconnected", uid)
        except Exception as exc:
            logger.exception("Error handling client %s", uid)
            if session:
                await session.send_status("ERROR", str(exc))
        finally:
            try:
                if websocket.client_state.name != "DISCONNECTED":
                    await websocket.close()
            except Exception:
                pass
            if uid and uid in self.sessions:
                del self.sessions[uid]
            logger.info("Client %s cleaned up", uid)

    async def _audio_loop(self, session: ClientSession):
        """Main audio processing loop for a client."""
        while session.connected:
            try:
                data = await session.websocket.receive_bytes()
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

            if session.use_vad:
                is_speech = await self._check_vad(session)
                if not is_speech:
                    continue

            await self._transcribe_if_ready(session)

    async def _check_vad(self, session: ClientSession) -> bool:
        """Check voice activity using VAD deployment."""
        if session.audio_buffer.shape[0] < VAD_FRAME_SIZE:
            return True

        frame = session.audio_buffer[-(VAD_FRAME_SIZE * 4) :]
        result = await self.vad_handle.detect.remote(frame)
        is_speech = result.get("is_speech", True)

        if not is_speech:
            session.no_voice_activity_chunks += 1
            if session.no_voice_activity_chunks > 3:
                session.eos = True
                await asyncio.sleep(0.1)
            return False
        else:
            session.no_voice_activity_chunks = 0
            session.eos = False
            return True

    async def _transcribe_if_ready(self, session: ClientSession):
        """Transcribe audio if enough data is available."""
        chunk, duration = session.get_audio_chunk()
        if chunk is None or duration < MIN_AUDIO_DURATION:
            return

        result = await self.transcriber_handle.transcribe.remote(
            audio=chunk,
            language=session.language,
            task=session.task,
            initial_prompt=session.initial_prompt,
        )

        if "error" in result:
            logger.error(f"Transcription error for {session.uid}: {result['error']}")
            return

        segments = result.get("segments", [])
        if not segments:
            session.update_offset(duration)
            return

        if session.language is None and result.get("language"):
            session.language = result["language"]
            await session.websocket.send_json(
                {
                    "uid": session.uid,
                    "language": result["language"],
                    "language_prob": result.get("language_probability", 0.0),
                }
            )

        response_segments = []
        last_completed_offset = 0.0
        for seg in segments[:-1]:
            if seg.get("no_speech_prob", 1.0) > session.no_speech_thresh:
                continue
            start = session.timestamp_offset + seg["start"]
            end = session.timestamp_offset + seg["end"]
            # Avoid re-sending segments we already completed
            if float(end) <= session.last_completed_end:
                continue
            response_segments.append(
                session.format_segment(start, end, seg["text"], completed=True)
            )
            session.transcript.append(
                session.format_segment(start, end, seg["text"], completed=True)
            )
            session.last_completed_end = float(end)
            last_completed_offset = seg["end"]

        # Advance timestamp_offset past completed segments so we don't
        # re-transcribe them on the next call
        if last_completed_offset > 0:
            session.update_offset(last_completed_offset)

        last_seg = segments[-1]
        if last_seg.get("no_speech_prob", 1.0) <= session.no_speech_thresh:
            session.current_out = last_seg["text"]
            start = session.timestamp_offset + last_seg["start"]
            end = session.timestamp_offset + last_seg["end"]
            response_segments.append(
                session.format_segment(start, end, session.current_out, completed=False)
            )

            if session.current_out.strip() == session.prev_out.strip():
                session.same_output_count += 1
            else:
                session.same_output_count = 0

            if session.same_output_count > session.same_output_threshold:
                session.transcript.append(
                    session.format_segment(
                        start, end, session.current_out, completed=True
                    )
                )
                session.last_completed_end = float(end)
                session.update_offset(last_seg["end"])
                session.current_out = ""
                session.same_output_count = 0
            else:
                session.prev_out = session.current_out
        else:
            session.update_offset(duration)

        if response_segments:
            await session.send_response(response_segments)

    async def _process_remaining(self, session: ClientSession):
        """Process any remaining audio in buffer."""
        chunk, duration = session.get_audio_chunk()
        if chunk is not None and duration > 0:
            result = await self.transcriber_handle.transcribe.remote(
                audio=chunk,
                language=session.language,
                task=session.task,
                initial_prompt=session.initial_prompt,
            )

            segments = result.get("segments", [])
            response_segments = []
            for seg in segments:
                start = session.timestamp_offset + seg["start"]
                end = session.timestamp_offset + seg["end"]
                response_segments.append(
                    session.format_segment(start, end, seg["text"], completed=True)
                )

            if response_segments:
                await session.send_response(response_segments)
