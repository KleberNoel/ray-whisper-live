# Ray Serve WhisperLive

Minimal Ray Serve implementation of [WhisperLive](https://github.com/collabora/WhisperLive) for real-time streaming transcription using `whisper-large-v3-turbo`.

## Architecture

Two Ray Serve deployments (VAD is inlined using `faster_whisper.vad`):

| Deployment | Role | Resources |
|---|---|---|
| `WhisperTranscriber` | faster-whisper `large-v3-turbo` on GPU | 1 CPU, 0.25 GPU |
| `WhisperLiveServer` | FastAPI WebSocket ingress + inline Silero VAD | 1 CPU, 0 GPU |

## Setup

```bash
conda activate ray-whisper-live
pip install -r requirements.txt
```

Set CUDA library paths before starting:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
```

## Run Server

```bash
python serve.py
```

Options:
- `--model-size` -- Whisper model size (default: `large-v3-turbo`)

Health check: `curl http://127.0.0.1:8000/health`

## Run Client

```bash
python client.py audio.wav
```

Options:
- `--host` -- Server host (default: `localhost`)
- `--port` -- Server port (default: `8000`)
- `--language` -- Language code (default: auto-detect)
- `--chunk-duration` -- Audio chunk duration in seconds (default: `0.5`)
- `--beam-size` -- Beam width (server default: `5`)
- `--no-speech-threshold` -- No-speech filter threshold (server default: `0.45`)
- `--vad-threshold` -- VAD speech probability threshold (server default: `0.5`)
- `--initial-prompt` -- Decoder prompt for context priming

## Benchmark

```bash
python benchmark.py --file audio.wav --reference "expected transcription text"
```

Outputs RTF (real-time factor) and WER (word error rate).

## WebSocket Protocol

Connect to `ws://<host>:8000/listen`

### 1. Send JSON options

All fields except `uid` are optional and fall back to server defaults.

```json
{
  "uid": "unique-id",
  "language": "en",
  "task": "transcribe",
  "use_vad": true,
  "initial_prompt": null,
  "beam_size": 5,
  "no_speech_threshold": 0.45,
  "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
  "condition_on_previous_text": true,
  "vad_threshold": 0.5,
  "min_silence_duration_ms": 2000,
  "speech_pad_ms": 400
}
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `uid` | str | `"unknown"` | Unique session identifier |
| `language` | str/null | `null` | BCP-47 code, `null` = auto-detect |
| `task` | str | `"transcribe"` | Task type |
| `use_vad` | bool | `true` | Gate transcription on voice activity |
| `initial_prompt` | str/null | `null` | Decoder prompt for context priming |
| `beam_size` | int | `5` | Beam width (higher = more accurate) |
| `no_speech_threshold` | float | `0.45` | Discard segments above this |
| `temperature` | float/list | `[0,.2,.4,.6,.8,1]` | Temperature fallback schedule |
| `condition_on_previous_text` | bool | `true` | Use previous output as context |
| `vad_threshold` | float | `0.5` | Silero speech probability threshold |
| `min_silence_duration_ms` | int | `2000` | Min silence to split speech (ms) |
| `speech_pad_ms` | int | `400` | Padding around detected speech (ms) |

### 2. Send audio chunks as binary (16 kHz float32 PCM)

### 3. Send `b"END_OF_AUDIO"` to signal end of stream

### 4. Receive JSON responses

```json
{
  "uid": "unique-id",
  "segments": [
    {"start": "0.000", "end": "1.234", "text": "Hello world", "completed": true}
  ]
}
```
