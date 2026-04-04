from src.config import (
    DISCARD_SECONDS,
    MAX_BUFFER_SECONDS,
    MIN_AUDIO_DURATION,
    SAME_OUTPUT_THRESHOLD,
    SAMPLE_RATE,
    AsrConfig,
)


class TestAsrConfigDefaults:
    """Verify AsrConfig default values."""

    def test_beam_size(self) -> None:
        cfg = AsrConfig()
        assert cfg.beam_size == 5

    def test_no_speech_threshold(self) -> None:
        cfg = AsrConfig()
        assert cfg.no_speech_threshold == 0.45

    def test_temperature_schedule(self) -> None:
        cfg = AsrConfig()
        assert cfg.temperature == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def test_condition_on_previous_text(self) -> None:
        cfg = AsrConfig()
        assert cfg.condition_on_previous_text is True

    def test_custom_values(self) -> None:
        cfg = AsrConfig(beam_size=3, no_speech_threshold=0.6)
        assert cfg.beam_size == 3
        assert cfg.no_speech_threshold == 0.6

    def test_temperature_independence(self) -> None:
        """Each instance should have its own temperature list."""
        a = AsrConfig()
        b = AsrConfig()
        a.temperature.append(1.5)
        assert 1.5 not in b.temperature


class TestConstants:
    """Verify module-level constants."""

    def test_sample_rate(self) -> None:
        assert SAMPLE_RATE == 16000

    def test_min_audio_duration(self) -> None:
        assert MIN_AUDIO_DURATION == 1.0

    def test_max_buffer_seconds(self) -> None:
        assert MAX_BUFFER_SECONDS == 45

    def test_discard_seconds(self) -> None:
        assert DISCARD_SECONDS == 30

    def test_same_output_threshold(self) -> None:
        assert SAME_OUTPUT_THRESHOLD == 10
