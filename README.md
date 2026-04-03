# Ray Serve WhisperLive

Minimal Ray Serve implementation of [WhisperLive](https://github.com/collabora/WhisperLive) for real-time streaming transcription using `whisper-large-v3-turbo`.

## Architecture

Three separate Ray Serve deployments:

| Deployment | Model | Resources | Purpose |
|---|---|---|---|
| `VADDetector` | Silero VAD (ONNX) | 0.5 CPU, 0 GPU | Voice activity detection |
| `WhisperTranscriber` | whisper-large-v3-turbo | 1 CPU, 0.25 GPU | High-accuracy transcription |
| `WhisperLiveServer` | FastAPI WebSocket | 1 CPU, 0 GPU | Client orchestration |

## Setup

```bash
conda env create -f environment.yml
conda activate ray-whisper-live
```

## Run Server

```bash
python serve.py
```

Options:
- `--vad-threshold` — VAD speech probability threshold (default: 0.5)
- `--model-size` — Whisper model size (default: large-v3-turbo)
- `--beam-size` — Beam size for decoding (default: 5)
- `--no-speech-threshold` — Silent segment filter threshold (default: 0.45)

## Run Client

```bash
python client.py audio.wav
```

Options:
- `--host` — Server host (default: localhost)
- `--port` — Server port (default: 8000)
- `--language` — Language code (default: auto-detect)
- `--task` — Task type: transcribe or translate (default: transcribe)
- `--chunk-duration` — Audio chunk duration in seconds (default: 0.5)

## WebSocket Protocol

Connect to `ws://<host>:8000/listen`

1. Send JSON options:
```json
{
  "uid": "unique-id",
  "language": "en",
  "task": "transcribe",
  "use_vad": true
}
```

2. Send audio chunks as binary (16kHz float32 PCM)

3. Send `b"END_OF_AUDIO"` to signal end of stream

4. Receive JSON responses:
```json
{
  "uid": "unique-id",
  "segments": [
    {
      "start": "0.000",
      "end": "1.234",
      "text": "Hello world",
      "completed": true
    }
  ]
}
```

## Accuracy Settings

- **Model**: `openai/whisper-large-v3-turbo` (best speed/accuracy tradeoff)
- **Compute type**: `float16` on CUDA
- **Decoding**: `beam_size=5`, temperature fallback `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`
- **VAD threshold**: `0.5` (filters silence before transcription)
- **No-speech threshold**: `0.45` (discards silent segments)
- **Condition on previous text**: `True` (context continuity)
