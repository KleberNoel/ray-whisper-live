import logging

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper.vad import VadOptions, get_speech_timestamps
from ray import serve

from src.config import MIN_AUDIO_DURATION, SAME_OUTPUT_THRESHOLD, AsrConfig
from src.session import ClientSession

logger = logging.getLogger(__name__)

app = FastAPI()


@serve.deployment(
    name="WhisperLiveServer",
    num_replicas=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0},
)
@serve.ingress(app)
class WhisperLiveServer:
    """WebSocket ingress that orchestrates Silero VAD, language detection,
    and Whisper ASR.

    VAD runs inline (CPU, via ``faster_whisper.vad``).  Language detection
    is delegated to :class:`LanguageDetector` and transcription to
    :class:`WhisperTranscriber`, both over Ray.

    Parameters
    ----------
    transcriber_handle : object
        Ray Serve handle for the transcriber deployment.
    language_detector_handle : object
        Ray Serve handle for the language detector deployment.
    """

    def __init__(  # noqa: ANN001
        self,
        transcriber_handle,
        language_detector_handle,
    ) -> None:
        self.transcriber_handle = transcriber_handle
        self.language_detector_handle = language_detector_handle
        self.sessions: dict[str, ClientSession] = {}

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
                "Client %s connected (vad=%s, beam=%d, no_speech=%.2f, lang=%s)",
                uid,
                session.use_vad,
                asr_cfg.beam_size,
                asr_cfg.no_speech_threshold,
                session.language or "auto",
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
        """Receive audio frames and transcribe when ready.

        Parameters
        ----------
        session : ClientSession
            The active client session.
        """
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

            # Detect language on first chunk with speech if not specified
            if not session.language_detected:
                await self._detect_language(session)

            await self._transcribe_if_ready(session)

    def _has_speech(self, session: ClientSession) -> bool:
        """Check the buffer tail for speech using Silero VAD.

        Parameters
        ----------
        session : ClientSession
            The active client session.

        Returns
        -------
        bool
            ``True`` if speech is detected or the buffer is too small to check.
        """
        if session.audio_buffer.shape[0] < 512:
            return True
        tail = session.audio_buffer[-2048:]
        timestamps = get_speech_timestamps(tail, session.vad)
        if not timestamps:
            session.no_voice_activity_chunks += 1
            return False
        session.no_voice_activity_chunks = 0
        return True

    async def _detect_language(self, session: ClientSession) -> None:
        """Detect language using the LanguageDetector deployment.

        Called once per session when no language was specified by the client.
        Uses the current audio buffer for detection.

        Parameters
        ----------
        session : ClientSession
            The active client session.
        """
        chunk, duration = session.get_audio_chunk()
        if chunk is None or duration < MIN_AUDIO_DURATION:
            return

        result = await self.language_detector_handle.detect.remote(audio=chunk)

        if "error" in result:
            logger.error(
                "Language detection error for %s: %s", session.uid, result["error"]
            )
            return

        session.language = result["language"]
        session.language_detected = True
        logger.info(
            "Client %s: detected language=%s (prob=%.3f)",
            session.uid,
            result["language"],
            result.get("language_probability", 0.0),
        )

        await session.websocket.send_json(
            {
                "uid": session.uid,
                "language": result["language"],
                "language_prob": result.get("language_probability", 0.0),
            }
        )

    async def _call_transcriber(
        self, session: ClientSession, audio: np.ndarray
    ) -> dict:
        """Dispatch a transcription request to the Ray transcriber.

        Parameters
        ----------
        session : ClientSession
            The active client session.
        audio : np.ndarray
            Float32 audio chunk at 16 kHz.

        Returns
        -------
        dict
            Transcription result from the transcriber deployment.
        """
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
        """Transcribe when at least *MIN_AUDIO_DURATION* seconds are buffered.

        Parameters
        ----------
        session : ClientSession
            The active client session.
        """
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
        """Flush any un-transcribed audio left in the buffer.

        Parameters
        ----------
        session : ClientSession
            The active client session.
        """
        chunk, duration = session.get_audio_chunk()
        if chunk is None or duration <= 0:
            return

        # Detect language if still unknown before final transcription
        if not session.language_detected:
            await self._detect_language(session)

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
