"""Benchmark script: RTF and WER measurement."""

import asyncio
import json
import time
import uuid
import os

import numpy as np
import soundfile as sf
import websockets
from jiwer import wer


def load_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load and resample audio file to target sample rate."""
    audio, sr = sf.read(file_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import scipy.signal

        num_samples = int(len(audio) * target_sr / sr)
        audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
    return audio


async def transcribe_file(
    file_path: str,
    host: str = "localhost",
    port: int = 8000,
    language: str | None = None,
    task: str = "transcribe",
    chunk_duration: float = 0.5,
):
    uri = f"ws://{host}:{port}/listen"
    uid = str(uuid.uuid4())

    audio = load_audio(file_path)
    duration = len(audio) / 16000
    chunk_size = int(chunk_duration * 16000)

    completed_segments = []  # Only finalized segments
    last_partial = None  # The most recent partial (in-progress) segment

    def merge_segments(incoming):
        nonlocal completed_segments, last_partial

        for seg in incoming:
            if seg.get("completed", False):
                # Deduplicate: skip if identical to the last completed segment
                if completed_segments:
                    prev = completed_segments[-1]
                    if (
                        seg["start"] == prev["start"]
                        and seg["end"] == prev["end"]
                        and seg["text"] == prev["text"]
                    ):
                        continue
                completed_segments.append(seg)
                # Clear partial if it overlaps with the newly completed segment
                last_partial = None
            else:
                # Partials just replace each other — only keep the latest
                last_partial = seg

    async with websockets.connect(uri) as ws:
        options = {
            "uid": uid,
            "language": language,
            "task": task,
            "use_vad": True,
        }
        await ws.send(json.dumps(options))

        response = await ws.recv()
        msg = json.loads(response)
        assert msg.get("message") == "SERVER_READY", f"Server not ready: {msg}"

        start_time = time.time()
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            await ws.send(chunk.tobytes())

            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    msg = json.loads(response)
                    if "segments" in msg:
                        merge_segments(msg["segments"])
            except asyncio.TimeoutError:
                pass

            elapsed = time.time() - start_time
            sent = (i + chunk_size) / 16000
            if sent < duration:
                wait = max(0, sent - elapsed)
                await asyncio.sleep(wait)

        await ws.send(b"END_OF_AUDIO")

        try:
            while True:
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                msg = json.loads(response)
                if "segments" in msg:
                    merge_segments(msg["segments"])
        except websockets.exceptions.ConnectionClosed:
            pass
        except asyncio.TimeoutError:
            pass

    total_time = time.time() - start_time
    rtf = total_time / duration if duration > 0 else float("inf")

    # Build final segments: all completed + last partial (if any)
    final_segments = list(completed_segments)
    if last_partial:
        final_segments.append(last_partial)

    text = " ".join(seg["text"] for seg in completed_segments)
    return text, rtf, duration, total_time, final_segments


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Audio file path")
    parser.add_argument("--reference", default=None, help="Reference text for WER")
    parser.add_argument("--language", default=None)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"File: {args.file}")
    print(f"Reference: {args.reference}")

    text, rtf, duration, total_time, segments = await transcribe_file(
        file_path=args.file,
        host=args.host,
        port=args.port,
        language=args.language,
    )

    print(f"\nTranscription ({duration:.2f}s audio, {total_time:.2f}s wall time):")
    print(f"RTF: {rtf:.3f}")
    print(f"Segments: {len(segments)}")
    print(f"Text: {text}")

    if args.reference:
        import re

        def normalize(s: str) -> str:
            s = s.lower()
            s = re.sub(r"[^\w\s]", "", s)  # strip punctuation
            s = re.sub(r"\s+", " ", s).strip()
            return s

        raw_error = wer(args.reference, text)
        norm_error = wer(normalize(args.reference), normalize(text))
        print(f"\nWER (raw):        {raw_error:.4f}")
        print(f"WER (normalized): {norm_error:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
