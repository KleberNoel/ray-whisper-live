from unittest.mock import MagicMock

import numpy as np

from src.transcriber import WhisperTranscriber

# Access the underlying class from the Ray Serve Deployment wrapper
_TranscriberCls = WhisperTranscriber.func_or_class


class TestWhisperTranscriber:
    """Test WhisperTranscriber with mocked WhisperModel."""

    @staticmethod
    def _make_transcriber(mock_model: MagicMock) -> _TranscriberCls:
        """Create a WhisperTranscriber with a mocked model, bypassing __init__."""
        t = _TranscriberCls.__new__(_TranscriberCls)
        t.model = mock_model
        return t

    def test_transcribe_success(self) -> None:
        mock_seg = MagicMock()
        mock_seg.start = 0.0
        mock_seg.end = 1.5
        mock_seg.text = " Hello world "
        mock_seg.no_speech_prob = 0.1

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_seg], mock_info)

        t = self._make_transcriber(mock_model)
        result = t.transcribe(np.zeros(16000, dtype=np.float32), language="en")

        assert result["language"] == "en"
        assert result["language_probability"] == 0.95
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "Hello world"
        assert result["segments"][0]["start"] == 0.0
        assert result["segments"][0]["end"] == 1.5
        assert result["segments"][0]["no_speech_prob"] == 0.1

    def test_transcribe_empty(self) -> None:
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.9

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], mock_info)

        t = self._make_transcriber(mock_model)
        result = t.transcribe(np.zeros(16000, dtype=np.float32))

        assert result["segments"] == []
        assert result["language"] == "en"

    def test_transcribe_error(self) -> None:
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("GPU OOM")

        t = self._make_transcriber(mock_model)
        result = t.transcribe(np.zeros(16000, dtype=np.float32))

        assert "error" in result
        assert result["segments"] == []
        assert "GPU OOM" in result["error"]

    def test_transcribe_passes_parameters(self) -> None:
        mock_info = MagicMock()
        mock_info.language = "fr"
        mock_info.language_probability = 0.8

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], mock_info)

        t = self._make_transcriber(mock_model)
        audio = np.zeros(16000, dtype=np.float32)
        t.transcribe(
            audio,
            language="fr",
            beam_size=3,
            no_speech_threshold=0.6,
            temperature=[0.0],
            condition_on_previous_text=False,
            initial_prompt="test prompt",
        )

        mock_model.transcribe.assert_called_once_with(
            audio,
            language="fr",
            task="transcribe",
            beam_size=3,
            temperature=[0.0],
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            initial_prompt="test prompt",
            vad_filter=False,
            word_timestamps=False,
            suppress_blank=True,
            suppress_tokens=[-1],
        )

    def test_default_temperatures(self) -> None:
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.9

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], mock_info)

        t = self._make_transcriber(mock_model)
        t.transcribe(np.zeros(16000, dtype=np.float32))

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["temperature"] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
