from unittest.mock import MagicMock

import numpy as np

from src.language_detector import LanguageDetector

# Access the underlying class from the Ray Serve Deployment wrapper
_DetectorCls = LanguageDetector.func_or_class


class TestLanguageDetector:
    """Test LanguageDetector with mocked WhisperModel."""

    @staticmethod
    def _make_detector(mock_model: MagicMock) -> _DetectorCls:
        """Create a LanguageDetector with a mocked model, bypassing __init__."""
        d = _DetectorCls.__new__(_DetectorCls)
        d.model = mock_model
        return d

    def test_detect_success(self) -> None:
        mock_model = MagicMock()
        mock_model.detect_language.return_value = (
            "en",
            0.95,
            [("en", 0.95), ("fr", 0.03)],
        )

        d = self._make_detector(mock_model)
        result = d.detect(np.zeros(16000, dtype=np.float32))

        assert result["language"] == "en"
        assert result["language_probability"] == 0.95

    def test_detect_french(self) -> None:
        mock_model = MagicMock()
        mock_model.detect_language.return_value = (
            "fr",
            0.88,
            [("fr", 0.88), ("en", 0.05)],
        )

        d = self._make_detector(mock_model)
        result = d.detect(np.zeros(16000, dtype=np.float32))

        assert result["language"] == "fr"
        assert result["language_probability"] == 0.88

    def test_detect_error(self) -> None:
        mock_model = MagicMock()
        mock_model.detect_language.side_effect = RuntimeError("Model error")

        d = self._make_detector(mock_model)
        result = d.detect(np.zeros(16000, dtype=np.float32))

        assert "error" in result
        assert "Model error" in result["error"]

    def test_detect_passes_audio(self) -> None:
        mock_model = MagicMock()
        mock_model.detect_language.return_value = ("en", 0.9, [])

        d = self._make_detector(mock_model)
        audio = np.ones(32000, dtype=np.float32)
        d.detect(audio)

        mock_model.detect_language.assert_called_once_with(audio)
