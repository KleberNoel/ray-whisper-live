import numpy as np
import pytest
from unittest.mock import AsyncMock

from src.config import DISCARD_SECONDS, MAX_BUFFER_SECONDS, SAMPLE_RATE
from src.session import ClientSession


class TestClientSessionInit:
    """Verify ClientSession initialization."""

    def test_defaults(self, mock_websocket: AsyncMock) -> None:
        s = ClientSession("u1", mock_websocket)
        assert s.uid == "u1"
        assert s.language is None
        assert s.task == "transcribe"
        assert s.use_vad is True
        assert s.connected is True
        assert s.language_detected is False
        assert s.audio_buffer.shape == (0,)

    def test_with_language(self, mock_websocket: AsyncMock) -> None:
        s = ClientSession("u2", mock_websocket, language="en")
        assert s.language == "en"
        assert s.language_detected is True

    def test_initial_offsets(self, mock_websocket: AsyncMock) -> None:
        s = ClientSession("u3", mock_websocket)
        assert s.timestamp_offset == 0.0
        assert s.frames_offset == 0.0
        assert s.last_completed_end == -1.0


class TestAddFrames:
    """Verify audio buffer management."""

    def test_append_to_empty(
        self, mock_websocket: AsyncMock, silence_1s: np.ndarray
    ) -> None:
        s = ClientSession("u1", mock_websocket)
        s.add_frames(silence_1s)
        assert s.audio_buffer.shape[0] == SAMPLE_RATE

    def test_concatenate(
        self, mock_websocket: AsyncMock, silence_1s: np.ndarray
    ) -> None:
        s = ClientSession("u1", mock_websocket)
        s.add_frames(silence_1s)
        s.add_frames(silence_1s)
        assert s.audio_buffer.shape[0] == 2 * SAMPLE_RATE

    def test_trim_on_overflow(self, mock_websocket: AsyncMock) -> None:
        s = ClientSession("u1", mock_websocket)
        # Add MAX_BUFFER_SECONDS + 1 second of audio
        big_chunk = np.zeros((MAX_BUFFER_SECONDS + 1) * SAMPLE_RATE, dtype=np.float32)
        s.add_frames(big_chunk)
        expected = (MAX_BUFFER_SECONDS + 1 - DISCARD_SECONDS) * SAMPLE_RATE
        assert s.audio_buffer.shape[0] == expected
        assert s.frames_offset == DISCARD_SECONDS

    def test_timestamp_offset_preserved_on_trim(
        self, mock_websocket: AsyncMock
    ) -> None:
        s = ClientSession("u1", mock_websocket)
        s.timestamp_offset = 40.0  # Already transcribed up to 40s
        big_chunk = np.zeros((MAX_BUFFER_SECONDS + 1) * SAMPLE_RATE, dtype=np.float32)
        s.add_frames(big_chunk)
        # timestamp_offset should stay at 40 since it's > frames_offset (30)
        assert s.timestamp_offset == 40.0

    def test_timestamp_offset_bumped_on_trim(self, mock_websocket: AsyncMock) -> None:
        s = ClientSession("u1", mock_websocket)
        s.timestamp_offset = 5.0  # Behind frames_offset after trim
        big_chunk = np.zeros((MAX_BUFFER_SECONDS + 1) * SAMPLE_RATE, dtype=np.float32)
        s.add_frames(big_chunk)
        # timestamp_offset should be bumped to frames_offset (30)
        assert s.timestamp_offset == DISCARD_SECONDS


class TestGetAudioChunk:
    """Verify un-transcribed tail extraction."""

    def test_empty_buffer(self, mock_websocket: AsyncMock) -> None:
        s = ClientSession("u1", mock_websocket)
        chunk, dur = s.get_audio_chunk()
        assert chunk is None
        assert dur == 0.0

    def test_full_buffer_returned(
        self, mock_websocket: AsyncMock, tone_1s: np.ndarray
    ) -> None:
        s = ClientSession("u1", mock_websocket)
        s.add_frames(tone_1s)
        chunk, dur = s.get_audio_chunk()
        assert chunk is not None
        assert chunk.shape[0] == SAMPLE_RATE
        assert abs(dur - 1.0) < 0.01

    def test_partial_after_offset(
        self, mock_websocket: AsyncMock, tone_1s: np.ndarray
    ) -> None:
        s = ClientSession("u1", mock_websocket)
        two_sec = np.concatenate([tone_1s, tone_1s])
        s.add_frames(two_sec)
        s.timestamp_offset = 1.0  # Already transcribed 1s
        chunk, dur = s.get_audio_chunk()
        assert chunk is not None
        assert chunk.shape[0] == SAMPLE_RATE
        assert abs(dur - 1.0) < 0.01

    def test_nothing_new(self, mock_websocket: AsyncMock, tone_1s: np.ndarray) -> None:
        s = ClientSession("u1", mock_websocket)
        s.add_frames(tone_1s)
        s.timestamp_offset = 1.0
        chunk, dur = s.get_audio_chunk()
        assert chunk is None
        assert dur == 0.0


class TestSendResponse:
    """Verify JSON response sending."""

    @pytest.mark.asyncio
    async def test_sends_json(self, mock_websocket: AsyncMock) -> None:
        s = ClientSession("u1", mock_websocket)
        segments = [{"start": "0.000", "end": "1.000", "text": "hi", "completed": True}]
        await s.send_response(segments)
        mock_websocket.send_json.assert_called_once_with(
            {"uid": "u1", "segments": segments}
        )

    @pytest.mark.asyncio
    async def test_no_send_when_disconnected(self, mock_websocket: AsyncMock) -> None:
        s = ClientSession("u1", mock_websocket)
        s.connected = False
        await s.send_response([])
        mock_websocket.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_marks_disconnected_on_error(self, mock_websocket: AsyncMock) -> None:
        s = ClientSession("u1", mock_websocket)
        mock_websocket.send_json.side_effect = ConnectionError("broken")
        await s.send_response([{"text": "hi"}])
        assert s.connected is False
