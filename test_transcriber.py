"""Direct test for the Whisper transcriber deployment logic."""

import argparse
import time

import numpy as np
import soundfile as sf

from transcriber_deployment import WhisperTranscriber


def load_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
    audio, sr = sf.read(file_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import scipy.signal

        num_samples = int(len(audio) * target_sr / sr)
        audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--language", default=None)
    parser.add_argument("--model", default="large-v3-turbo")
    args = parser.parse_args()

    audio = load_audio(args.file)
    duration = len(audio) / 16000.0

    transcriber = WhisperTranscriber(model_size=args.model)

    started = time.perf_counter()
    result = transcriber.transcribe(audio=audio, language=args.language)
    elapsed = time.perf_counter() - started

    text = " ".join(seg["text"] for seg in result.get("segments", []))
    rtf = elapsed / duration if duration else float("inf")

    print(f"duration_s={duration:.3f}")
    print(f"elapsed_s={elapsed:.3f}")
    print(f"rtf={rtf:.3f}")
    print(f"language={result.get('language')}")
    print("text=")
    print(text)


if __name__ == "__main__":
    main()
