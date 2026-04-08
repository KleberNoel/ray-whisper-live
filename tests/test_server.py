from unittest.mock import AsyncMock

import numpy as np
import pytest
from faster_whisper.vad import VadOptions

from src.config import MIN_AUDIO_DURATION, SAMPLE_RATE, AsrConfig
from src.server import WhisperLiveServer
from src.session import ClientSession

# Access the underlying class from the Ray Serve Deployment wrapper
_ServerCls = WhisperLiveServer.func_or_class


def _make_server(
    transcriber_handle: AsyncMock | None = None,
    vad_handle: AsyncMock | None = None,
) -> _ServerCls:
    """Create a WhisperLiveServer with mocked handles, bypassing __init__."""
    s = _ServerCls.__new__(_ServerCls)
    s.transcriber_handle = transcriber_handle or AsyncMock()
    s.vad_handle = vad_handle or AsyncMock()
    s.sessions = {}
    return s


def _make_session(
    mock_websocket: AsyncMock,
    uid: str = "test-uid",
    use_vad: bool = True,
    language: str | None = None,
) -> ClientSession:
    """Create a ClientSession with defaults suitable for testing."""
    return ClientSession(
        uid,
        mock_websocket,
        language=language,
        use_vad=use_vad,
        vad=VadOptions(threshold=0.5, min_silence_duration_ms=2000, speech_pad_ms=400),
        asr=AsrConfig(),
    )


class TestHasSpeech:
    """Verify _has_speech delegates to vad_handle."""

    @pytest.mark.asyncio
    async def test_calls_vad_handle(self, mock_websocket: AsyncMock) -> None:
        """Verify vad_handle.has_speech.remote is called with correct args."""
        vad_handle = AsyncMock()
        vad_handle.has_speech.remote = AsyncMock(return_value=True)
        server = _make_server(vad_handle=vad_handle)
        session = _make_session(mock_websocket)

        # Fill buffer with enough audio (>= 512 samples)
        session.add_frames(np.zeros(2048, dtype=np.float32))

        result = await server._has_speech(session)

        assert result is True
        vad_handle.has_speech.remote.assert_called_once()
        call_kwargs = vad_handle.has_speech.remote.call_args.kwargs
        assert call_kwargs["threshold"] == 0.5
        assert call_kwargs["min_silence_duration_ms"] == 2000
        assert call_kwargs["speech_pad_ms"] == 400
        assert call_kwargs["audio"].shape[0] == 2048

    @pytest.mark.asyncio
    async def test_returns_true_for_short_buffer(
        self, mock_websocket: AsyncMock
    ) -> None:
        """Buffer < 512 samples should skip VAD and return True."""
        vad_handle = AsyncMock()
        server = _make_server(vad_handle=vad_handle)
        session = _make_session(mock_websocket)

        session.add_frames(np.zeros(256, dtype=np.float32))
        result = await server._has_speech(session)

        assert result is True
        vad_handle.has_speech.remote.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_speech_increments_counter(
        self, mock_websocket: AsyncMock
    ) -> None:
        """When VAD says no speech, no_voice_activity_chunks should increment."""
        vad_handle = AsyncMock()
        vad_handle.has_speech.remote = AsyncMock(return_value=False)
        server = _make_server(vad_handle=vad_handle)
        session = _make_session(mock_websocket)
        session.add_frames(np.zeros(2048, dtype=np.float32))

        result = await server._has_speech(session)

        assert result is False
        assert session.no_voice_activity_chunks == 1

    @pytest.mark.asyncio
    async def test_speech_resets_counter(self, mock_websocket: AsyncMock) -> None:
        """When VAD detects speech, no_voice_activity_chunks should reset to 0."""
        vad_handle = AsyncMock()
        vad_handle.has_speech.remote = AsyncMock(return_value=True)
        server = _make_server(vad_handle=vad_handle)
        session = _make_session(mock_websocket)
        session.add_frames(np.zeros(2048, dtype=np.float32))
        session.no_voice_activity_chunks = 5  # Was high

        result = await server._has_speech(session)

        assert result is True
        assert session.no_voice_activity_chunks == 0

    @pytest.mark.asyncio
    async def test_passes_tail_2048(self, mock_websocket: AsyncMock) -> None:
        """Should pass the last 2048 samples even if buffer is larger."""
        vad_handle = AsyncMock()
        vad_handle.has_speech.remote = AsyncMock(return_value=True)
        server = _make_server(vad_handle=vad_handle)
        session = _make_session(mock_websocket)

        # Add 8000 samples — should still only pass last 2048
        session.add_frames(np.ones(8000, dtype=np.float32))

        await server._has_speech(session)

        call_kwargs = vad_handle.has_speech.remote.call_args.kwargs
        assert call_kwargs["audio"].shape[0] == 2048


class TestAudioLoop:
    """Verify _audio_loop wiring: frames -> VAD -> transcribe."""

    @pytest.mark.asyncio
    async def test_vad_called_on_each_audio_frame(
        self, mock_websocket: AsyncMock
    ) -> None:
        """Each audio frame should trigger a VAD check via vad_handle."""
        vad_handle = AsyncMock()
        # VAD says speech on all frames
        vad_handle.has_speech.remote = AsyncMock(return_value=True)

        transcriber_handle = AsyncMock()
        transcriber_handle.transcribe.remote = AsyncMock(return_value={"segments": []})

        server = _make_server(
            transcriber_handle=transcriber_handle, vad_handle=vad_handle
        )
        session = _make_session(mock_websocket)

        # Simulate 3 audio frames, then END_OF_AUDIO
        frame = np.zeros(1024, dtype=np.float32)
        frames_sent = 3
        call_count = 0

        async def fake_receive_bytes() -> bytes:
            nonlocal call_count
            call_count += 1
            if call_count <= frames_sent:
                return frame.tobytes()
            return b"END_OF_AUDIO"

        session.websocket.receive_bytes = fake_receive_bytes

        await server._audio_loop(session)

        # VAD should be called for frames 2 and 3 (frame 1 has < 512 samples)
        # Frame 1: 1024 samples total -> >= 512, so VAD called
        # Frame 2: 2048 samples total -> VAD called
        # Frame 3: 3072 samples total -> VAD called
        assert vad_handle.has_speech.remote.call_count == frames_sent

    @pytest.mark.asyncio
    async def test_vad_false_skips_transcription(
        self, mock_websocket: AsyncMock
    ) -> None:
        """When VAD returns False, transcriber should not be called."""
        vad_handle = AsyncMock()
        vad_handle.has_speech.remote = AsyncMock(return_value=False)

        transcriber_handle = AsyncMock()
        transcriber_handle.transcribe.remote = AsyncMock(return_value={"segments": []})

        server = _make_server(
            transcriber_handle=transcriber_handle, vad_handle=vad_handle
        )
        session = _make_session(mock_websocket)

        # Send enough audio to exceed MIN_AUDIO_DURATION, then END_OF_AUDIO
        big_frame = np.zeros(int(SAMPLE_RATE * 2), dtype=np.float32)
        call_count = 0

        async def fake_receive_bytes() -> bytes:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return big_frame.tobytes()
            return b"END_OF_AUDIO"

        session.websocket.receive_bytes = fake_receive_bytes

        await server._audio_loop(session)

        # VAD said no speech, so transcriber should only be called in
        # _process_remaining (END_OF_AUDIO flush), not in the main loop
        vad_handle.has_speech.remote.assert_called_once()
        # _process_remaining does call the transcriber for flush
        assert transcriber_handle.transcribe.remote.call_count == 1

    @pytest.mark.asyncio
    async def test_vad_true_triggers_transcription(
        self, mock_websocket: AsyncMock
    ) -> None:
        """When VAD returns True and enough audio is buffered, transcriber is called."""
        vad_handle = AsyncMock()
        vad_handle.has_speech.remote = AsyncMock(return_value=True)

        transcriber_handle = AsyncMock()
        transcriber_handle.transcribe.remote = AsyncMock(return_value={"segments": []})

        server = _make_server(
            transcriber_handle=transcriber_handle, vad_handle=vad_handle
        )
        session = _make_session(mock_websocket)

        # Send audio > MIN_AUDIO_DURATION (1.0s), then END_OF_AUDIO
        big_frame = np.zeros(int(SAMPLE_RATE * 2), dtype=np.float32)
        call_count = 0

        async def fake_receive_bytes() -> bytes:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return big_frame.tobytes()
            return b"END_OF_AUDIO"

        session.websocket.receive_bytes = fake_receive_bytes

        await server._audio_loop(session)

        vad_handle.has_speech.remote.assert_called_once()
        # Transcriber called in the loop (speech detected + enough audio)
        # plus _process_remaining on END_OF_AUDIO
        assert transcriber_handle.transcribe.remote.call_count >= 1

    @pytest.mark.asyncio
    async def test_vad_disabled_skips_vad_handle(
        self, mock_websocket: AsyncMock
    ) -> None:
        """When use_vad=False, vad_handle should never be called."""
        vad_handle = AsyncMock()
        vad_handle.has_speech.remote = AsyncMock(return_value=True)

        transcriber_handle = AsyncMock()
        transcriber_handle.transcribe.remote = AsyncMock(return_value={"segments": []})

        server = _make_server(
            transcriber_handle=transcriber_handle, vad_handle=vad_handle
        )
        session = _make_session(mock_websocket, use_vad=False)

        frame = np.zeros(int(SAMPLE_RATE * 2), dtype=np.float32)
        call_count = 0

        async def fake_receive_bytes() -> bytes:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return frame.tobytes()
            return b"END_OF_AUDIO"

        session.websocket.receive_bytes = fake_receive_bytes

        await server._audio_loop(session)

        vad_handle.has_speech.remote.assert_not_called()


class TestProcessRemaining:
    """Verify _process_remaining flushes without VAD."""

    @pytest.mark.asyncio
    async def test_flush_calls_transcriber_not_vad(
        self, mock_websocket: AsyncMock
    ) -> None:
        """END_OF_AUDIO flush should call transcriber directly, no VAD check."""
        vad_handle = AsyncMock()
        transcriber_handle = AsyncMock()
        transcriber_handle.transcribe.remote = AsyncMock(
            return_value={
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "hello", "no_speech_prob": 0.1}
                ]
            }
        )

        server = _make_server(
            transcriber_handle=transcriber_handle, vad_handle=vad_handle
        )
        session = _make_session(mock_websocket)
        session.add_frames(np.zeros(SAMPLE_RATE * 2, dtype=np.float32))

        await server._process_remaining(session)

        transcriber_handle.transcribe.remote.assert_called_once()
        # VAD is not involved in _process_remaining
        vad_handle.has_speech.remote.assert_not_called()
        # Response should have been sent
        mock_websocket.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_with_empty_buffer_is_noop(
        self, mock_websocket: AsyncMock
    ) -> None:
        """Empty buffer should cause no transcriber call."""
        transcriber_handle = AsyncMock()
        server = _make_server(transcriber_handle=transcriber_handle)
        session = _make_session(mock_websocket)

        await server._process_remaining(session)

        transcriber_handle.transcribe.remote.assert_not_called()


class TestTranscribeIfReady:
    """Verify _transcribe_if_ready sends segments to client."""

    @pytest.mark.asyncio
    async def test_sends_segments_to_client(self, mock_websocket: AsyncMock) -> None:
        transcriber_handle = AsyncMock()
        transcriber_handle.transcribe.remote = AsyncMock(
            return_value={
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.5,
                        "text": " Hello world",
                        "no_speech_prob": 0.1,
                    }
                ]
            }
        )

        server = _make_server(transcriber_handle=transcriber_handle)
        session = _make_session(mock_websocket)
        session.add_frames(np.zeros(SAMPLE_RATE * 2, dtype=np.float32))

        await server._transcribe_if_ready(session)

        transcriber_handle.transcribe.remote.assert_called_once()
        mock_websocket.send_json.assert_called_once()
        sent = mock_websocket.send_json.call_args[0][0]
        assert sent["uid"] == "test-uid"
        assert len(sent["segments"]) == 1
        assert sent["segments"][0]["text"] == " Hello world"

    @pytest.mark.asyncio
    async def test_not_enough_audio_is_noop(self, mock_websocket: AsyncMock) -> None:
        """Less than MIN_AUDIO_DURATION should not trigger transcription."""
        transcriber_handle = AsyncMock()
        server = _make_server(transcriber_handle=transcriber_handle)
        session = _make_session(mock_websocket)

        # Add less than MIN_AUDIO_DURATION (1.0s) of audio
        session.add_frames(
            np.zeros(int(SAMPLE_RATE * MIN_AUDIO_DURATION * 0.5), dtype=np.float32)
        )

        await server._transcribe_if_ready(session)

        transcriber_handle.transcribe.remote.assert_not_called()
