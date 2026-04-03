import argparse
import asyncio
import json
import time
import uuid

import numpy as np
import soundfile as sf
import websockets


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


def _build_options(
    uid: str,
    *,
    language: str | None = None,
    task: str = "transcribe",
    use_vad: bool = True,
    beam_size: int | None = None,
    no_speech_threshold: float | None = None,
    vad_threshold: float | None = None,
    initial_prompt: str | None = None,
) -> dict:
    """Build the JSON options dict sent on WebSocket connect.

    Only includes keys whose values differ from the server defaults so
    that the server's own defaults apply when omitted.

    Parameters
    ----------
    uid : str
        Unique session identifier.
    language : str or None
        BCP-47 language code (``None`` = auto-detect).
    task : str
        ``"transcribe"`` (default).
    use_vad : bool
        Enable voice-activity gating.
    beam_size : int or None
        Override server default beam width.
    no_speech_threshold : float or None
        Override server default no-speech threshold.
    vad_threshold : float or None
        Override server default VAD threshold.
    initial_prompt : str or None
        Decoder prompt for context priming.

    Returns
    -------
    dict
        Ready-to-serialise options dict.
    """
    opts: dict = {"uid": uid, "language": language, "task": task, "use_vad": use_vad}
    if beam_size is not None:
        opts["beam_size"] = beam_size
    if no_speech_threshold is not None:
        opts["no_speech_threshold"] = no_speech_threshold
    if vad_threshold is not None:
        opts["vad_threshold"] = vad_threshold
    if initial_prompt is not None:
        opts["initial_prompt"] = initial_prompt
    return opts


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
    """Stream an audio file to the server and print live transcriptions.

    Parameters
    ----------
    file_path : str
        Path to the audio file.
    host : str
        Server hostname.
    port : int
        Server port.
    language : str or None
        Language code (``None`` = auto-detect).
    task : str
        ``"transcribe"`` (default).
    chunk_duration : float
        Duration of each sent chunk in seconds.
    beam_size : int or None
        Override beam width.
    no_speech_threshold : float or None
        Override no-speech threshold.
    vad_threshold : float or None
        Override VAD threshold.
    initial_prompt : str or None
        Decoder prompt.
    """
    uri = f"ws://{host}:{port}/listen"
    uid = str(uuid.uuid4())

    print(f"Loading audio: {file_path}")
    audio = load_audio(file_path)
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

            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    data = json.loads(response)
                    if "segments" in data:
                        for seg in data["segments"]:
                            marker = "[DONE]" if seg.get("completed") else "[...]"
                            print(
                                f"  {marker} [{seg['start']}s -> {seg['end']}s] {seg['text']}"
                            )
                    elif "status" in data:
                        print(f"  Status: {data['status']} - {data['message']}")
            except asyncio.TimeoutError:
                pass

            elapsed = time.time() - start_time
            sent = (i + chunk_size) / 16000
            if sent < duration:
                await asyncio.sleep(max(0, sent - elapsed))

        print("Sending END_OF_AUDIO...")
        await ws.send(b"END_OF_AUDIO")

        print("Waiting for final results...")
        try:
            while True:
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(response)
                if "segments" in data:
                    for seg in data["segments"]:
                        marker = "[DONE]" if seg.get("completed") else "[...]"
                        print(
                            f"  {marker} [{seg['start']}s -> {seg['end']}s] {seg['text']}"
                        )
                elif "status" in data:
                    print(f"  Status: {data['status']} - {data['message']}")
        except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError):
            pass

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
