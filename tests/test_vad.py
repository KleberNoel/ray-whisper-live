from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.vad import SileroVadDeployment

# Access the underlying class from the Ray Serve Deployment wrapper
_VadCls = SileroVadDeployment.func_or_class


class TestSileroVadDeployment:
    """Test SileroVadDeployment with mocked faster_whisper.vad functions."""

    @staticmethod
    def _make_vad(mock_model: MagicMock | None = None) -> _VadCls:
        """Create a SileroVadDeployment bypassing __init__."""
        v = _VadCls.__new__(_VadCls)
        v._model = mock_model or MagicMock()
        return v

    @pytest.mark.asyncio
    @patch("src.vad.get_speech_timestamps")
    async def test_has_speech_returns_true(self, mock_timestamps: MagicMock) -> None:
        mock_timestamps.return_value = [{"start": 0, "end": 1024}]
        v = self._make_vad()

        result = await v.has_speech(np.zeros(2048, dtype=np.float32))

        assert result is True
        mock_timestamps.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.vad.get_speech_timestamps")
    async def test_has_speech_returns_false(self, mock_timestamps: MagicMock) -> None:
        mock_timestamps.return_value = []
        v = self._make_vad()

        result = await v.has_speech(np.zeros(2048, dtype=np.float32))

        assert result is False

    @pytest.mark.asyncio
    @patch("src.vad.get_speech_timestamps")
    async def test_has_speech_short_audio_returns_true(
        self, mock_timestamps: MagicMock
    ) -> None:
        """Audio shorter than 512 samples should always return True."""
        v = self._make_vad()

        result = await v.has_speech(np.zeros(256, dtype=np.float32))

        assert result is True
        mock_timestamps.assert_not_called()

    @pytest.mark.asyncio
    @patch("src.vad.get_speech_timestamps")
    async def test_has_speech_passes_vad_options(
        self, mock_timestamps: MagicMock
    ) -> None:
        mock_timestamps.return_value = [{"start": 0, "end": 512}]
        v = self._make_vad()

        await v.has_speech(
            np.zeros(1024, dtype=np.float32),
            threshold=0.7,
            min_silence_duration_ms=500,
            speech_pad_ms=200,
        )

        call_args = mock_timestamps.call_args
        opts = call_args[0][1]  # Second positional arg is VadOptions
        assert opts.threshold == 0.7
        assert opts.min_silence_duration_ms == 500
        assert opts.speech_pad_ms == 200

    @pytest.mark.asyncio
    @patch("src.vad.get_speech_timestamps")
    async def test_get_speech_segments(self, mock_timestamps: MagicMock) -> None:
        expected = [{"start": 100, "end": 500}, {"start": 800, "end": 1200}]
        mock_timestamps.return_value = expected
        v = self._make_vad()

        result = await v.get_speech_segments(np.zeros(2048, dtype=np.float32))

        assert result == expected

    @pytest.mark.asyncio
    @patch("src.vad.get_speech_timestamps")
    async def test_get_speech_segments_empty(self, mock_timestamps: MagicMock) -> None:
        mock_timestamps.return_value = []
        v = self._make_vad()

        result = await v.get_speech_segments(np.zeros(2048, dtype=np.float32))

        assert result == []

    @patch("src.vad.get_vad_model")
    def test_init_loads_model(self, mock_get_model: MagicMock) -> None:
        mock_get_model.return_value = MagicMock()

        v = _VadCls()

        mock_get_model.assert_called_once()
        assert v._model is not None
