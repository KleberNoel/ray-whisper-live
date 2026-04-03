import argparse
import asyncio
import json
import re
import time
import uuid

import numpy as np
import soundfile as sf
import websockets
from jiwer import wer


def load_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load an audio file and resample to *target_sr* Hz mono float32.

    Parameters
    ----------
    file_path : str
        Path to any format supported by *libsndfile*.
    target_sr : int
        Target sample rate in Hz.

    Returns
    -------
    np.ndarray
        1-D float32 array normalised to ``[-1, 1]``.
    """
    audio, sr = sf.read(file_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import scipy.signal

        num_samples = int(len(audio) * target_sr / sr)
        audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
    return audio


def _normalize(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


async def transcribe_file(
    file_path: str,
    *,
    host: str = "localhost",
    port: int = 8000,
    language: str | None = None,
    task: str = "transcribe",
    chunk_duration: float = 0.5,
) -> tuple[str, float, float, float, list[dict]]:
    """Stream an audio file and collect completed segments.

    Parameters
    ----------
    file_path : str
        Path to the audio file.
    host : str
        Server hostname.
    port : int
        Server port.
    language : str or None
        BCP-47 language code (``None`` = auto-detect).
    task : str
        ``"transcribe"`` (default).
    chunk_duration : float
        Duration of each sent chunk in seconds.

    Returns
    -------
    tuple[str, float, float, float, list[dict]]
        ``(text, rtf, audio_duration, wall_time, segments)``.
    """
    uri = f"ws://{host}:{port}/listen"
    uid = str(uuid.uuid4())

    audio = load_audio(file_path)
    duration = len(audio) / 16000
    chunk_size = int(chunk_duration * 16000)

    completed_segments: list[dict] = []

    def _merge(incoming: list[dict]) -> None:
        for seg in incoming:
            if not seg.get("completed", False):
                continue
            if completed_segments:
                prev = completed_segments[-1]
                if (
                    seg["start"] == prev["start"]
                    and seg["end"] == prev["end"]
                    and seg["text"] == prev["text"]
                ):
                    continue
            completed_segments.append(seg)

    async with websockets.connect(uri) as ws:
        await ws.send(
            json.dumps(
                {"uid": uid, "language": language, "task": task, "use_vad": True}
            )
        )

        msg = json.loads(await ws.recv())
        assert msg.get("message") == "SERVER_READY", f"Server not ready: {msg}"

        start_time = time.time()
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            await ws.send(chunk.tobytes())

            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    data = json.loads(response)
                    if "segments" in data:
                        _merge(data["segments"])
            except asyncio.TimeoutError:
                pass

            elapsed = time.time() - start_time
            sent = (i + chunk_size) / 16000
            if sent < duration:
                await asyncio.sleep(max(0, sent - elapsed))

        await ws.send(b"END_OF_AUDIO")

        try:
            while True:
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(response)
                if "segments" in data:
                    _merge(data["segments"])
        except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError):
            pass

    total_time = time.time() - start_time
    rtf = total_time / duration if duration > 0 else float("inf")
    text = " ".join(seg["text"] for seg in completed_segments)
    return text, rtf, duration, total_time, completed_segments


async def main() -> None:
    parser = argparse.ArgumentParser(description="WhisperLive benchmark (RTF + WER)")
    parser.add_argument("--file", required=True, help="Audio file path")
    parser.add_argument("--reference", default=None, help="Reference text for WER")
    parser.add_argument("--language", default=None, help="Language code")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"File: {args.file}")
    text, rtf, duration, total_time, segments = await transcribe_file(
        args.file, host=args.host, port=args.port, language=args.language
    )

    print(f"\nTranscription ({duration:.2f}s audio, {total_time:.2f}s wall time):")
    print(f"RTF: {rtf:.3f}")
    print(f"Segments: {len(segments)}")
    print(f"Text: {text}")

    if args.reference:
        raw_error = wer(args.reference, text)
        norm_error = wer(_normalize(args.reference), _normalize(text))
        print(f"\nWER (raw):        {raw_error:.4f}")
        print(f"WER (normalized): {norm_error:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
