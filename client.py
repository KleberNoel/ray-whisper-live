"""Test client for Ray Serve WhisperLive."""

import argparse
import asyncio
import json
import time
import uuid

import numpy as np
import soundfile as sf
import websockets


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
    """
    Send audio file to WhisperLive server for transcription.

    Args:
        file_path: Path to audio file
        host: Server host
        port: Server port
        language: Language code (None for auto-detect)
        task: "transcribe" or "translate"
        chunk_duration: Duration of each audio chunk in seconds
    """
    uri = f"ws://{host}:{port}/listen"
    uid = str(uuid.uuid4())

    print(f"Loading audio: {file_path}")
    audio = load_audio(file_path)
    duration = len(audio) / 16000
    print(f"Audio duration: {duration:.2f}s")

    chunk_size = int(chunk_duration * 16000)

    print(f"Connecting to {uri}...")
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
        print(f"Server response: {msg}")

        if msg.get("message") != "SERVER_READY":
            print("Server not ready, exiting")
            return

        print("Sending audio chunks...")
        start_time = time.time()
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            audio_bytes = chunk.tobytes()
            await ws.send(audio_bytes)

            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    msg = json.loads(response)
                    if "segments" in msg:
                        for seg in msg["segments"]:
                            completed = seg.get("completed", False)
                            marker = "[DONE]" if completed else "[...]"
                            print(
                                f"  {marker} [{seg['start']}s -> {seg['end']}s] {seg['text']}"
                            )
                    elif "status" in msg:
                        print(f"  Status: {msg['status']} - {msg['message']}")
            except asyncio.TimeoutError:
                pass

            elapsed = time.time() - start_time
            sent = (i + chunk_size) / 16000
            if sent < duration:
                wait = max(0, sent - elapsed)
                await asyncio.sleep(wait)

        print("Sending END_OF_AUDIO...")
        await ws.send(b"END_OF_AUDIO")

        print("Waiting for final results...")
        try:
            while True:
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                msg = json.loads(response)
                if "segments" in msg:
                    for seg in msg["segments"]:
                        completed = seg.get("completed", False)
                        marker = "[DONE]" if completed else "[...]"
                        print(
                            f"  {marker} [{seg['start']}s -> {seg['end']}s] {seg['text']}"
                        )
                elif "status" in msg:
                    print(f"  Status: {msg['status']} - {msg['message']}")
        except websockets.exceptions.ConnectionClosed:
            pass
        except asyncio.TimeoutError:
            pass

    print("Done")


def main():
    parser = argparse.ArgumentParser(description="Ray Serve WhisperLive Client")
    parser.add_argument("file", help="Path to audio file")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--language", default=None, help="Language code")
    parser.add_argument("--task", default="transcribe", help="Task type")
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=0.5,
        help="Audio chunk duration in seconds",
    )
    args = parser.parse_args()

    asyncio.run(
        transcribe_file(
            file_path=args.file,
            host=args.host,
            port=args.port,
            language=args.language,
            task=args.task,
            chunk_duration=args.chunk_duration,
        )
    )


if __name__ == "__main__":
    main()
