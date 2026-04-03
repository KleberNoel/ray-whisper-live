import argparse
import asyncio
import json
import time
import uuid

import websockets
from faster_whisper import decode_audio


def _build_options(uid: str, **overrides: object) -> dict:
    """Build the JSON options dict sent on WebSocket connect.

    Keys with ``None`` values are omitted so server defaults apply.
    """
    opts: dict = {"uid": uid, "language": None, "task": "transcribe", "use_vad": True}
    opts.update({k: v for k, v in overrides.items() if v is not None})
    return opts


async def _drain_responses(
    ws: websockets.ClientConnection,
    *,
    timeout: float = 0.1,
) -> None:
    """Read and print all pending transcription responses."""
    try:
        while True:
            data = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
            if "segments" in data:
                for seg in data["segments"]:
                    marker = "[DONE]" if seg.get("completed") else "[...]"
                    print(
                        f"  {marker} [{seg['start']}s -> {seg['end']}s] {seg['text']}"
                    )
            elif "status" in data:
                print(f"  Status: {data['status']} - {data['message']}")
    except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
        pass


async def transcribe_file(
    file_path: str,
    *,
    host: str = "localhost",
    port: int = 8000,
    language: str | None = None,
    task: str = "transcribe",
    chunk_duration: float = 0.5,
    beam_size: int | None = None,
    no_speech_threshold: float | None = None,
    vad_threshold: float | None = None,
    initial_prompt: str | None = None,
) -> None:
    """Stream an audio file to the server and print live transcriptions."""
    uri = f"ws://{host}:{port}/listen"
    uid = str(uuid.uuid4())

    print(f"Loading audio: {file_path}")
    audio = decode_audio(file_path, sampling_rate=16000)
    duration = len(audio) / 16000
    print(f"Audio duration: {duration:.2f}s")

    chunk_size = int(chunk_duration * 16000)
    options = _build_options(
        uid,
        language=language,
        task=task,
        beam_size=beam_size,
        no_speech_threshold=no_speech_threshold,
        vad_threshold=vad_threshold,
        initial_prompt=initial_prompt,
    )

    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps(options))

        msg = json.loads(await ws.recv())
        print(f"Server response: {msg}")
        if msg.get("message") != "SERVER_READY":
            print("Server not ready, exiting")
            return

        print("Sending audio chunks...")
        start_time = time.time()
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            await ws.send(chunk.tobytes())
            await _drain_responses(ws, timeout=0.1)

            elapsed = time.time() - start_time
            sent = (i + chunk_size) / 16000
            if sent < duration:
                await asyncio.sleep(max(0, sent - elapsed))

        print("Sending END_OF_AUDIO...")
        await ws.send(b"END_OF_AUDIO")

        print("Waiting for final results...")
        await _drain_responses(ws, timeout=5.0)

    print("Done")


def main() -> None:
    parser = argparse.ArgumentParser(description="WhisperLive streaming client")
    parser.add_argument("file", help="Path to audio file")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--language", default=None, help="Language code (auto-detect if omitted)"
    )
    parser.add_argument("--task", default="transcribe", help="Task type")
    parser.add_argument(
        "--chunk-duration", type=float, default=0.5, help="Chunk duration in seconds"
    )
    parser.add_argument(
        "--beam-size", type=int, default=None, help="Beam width (server default: 5)"
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=None,
        help="No-speech threshold (server default: 0.45)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=None,
        help="VAD speech threshold (server default: 0.5)",
    )
    parser.add_argument(
        "--initial-prompt", default=None, help="Decoder prompt for context priming"
    )
    args = parser.parse_args()

    asyncio.run(
        transcribe_file(
            args.file,
            host=args.host,
            port=args.port,
            language=args.language,
            task=args.task,
            chunk_duration=args.chunk_duration,
            beam_size=args.beam_size,
            no_speech_threshold=args.no_speech_threshold,
            vad_threshold=args.vad_threshold,
            initial_prompt=args.initial_prompt,
        )
    )


if __name__ == "__main__":
    main()
