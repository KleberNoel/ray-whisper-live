from dataclasses import dataclass, field

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
