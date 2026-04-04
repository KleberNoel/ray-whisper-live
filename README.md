# Ray Serve WhisperLive

Real-time streaming speech transcription via WebSocket, powered by [Ray Serve](https://docs.ray.io/en/latest/serve/) and [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Architecture

Three Ray Serve deployments:

| Deployment | Model | Role | Resources |
|---|---|---|---|
| `WhisperTranscriber` | `large-v3-turbo` | Transcription (GPU) | 1 CPU, 0.25 GPU |
| `LanguageDetector` | `large-v3` | Language detection (GPU) | 1 CPU, 0.25 GPU |
| `WhisperLiveServer` | -- | WebSocket ingress + inline Silero VAD | 1 CPU, 0 GPU |

When no language is specified by the client, `LanguageDetector` identifies the language on the first speech chunk, then `WhisperTranscriber` handles all transcription.

## Setup

```bash
conda activate ray-whisper-live
pip install -r requirements.txt
```

Set CUDA library paths:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
```

## Usage

```bash
# Start server
python serve.py

# Stream audio file
python client.py audio.wav

# Benchmark (RTF + WER)
python benchmark.py --file audio.wav --reference "expected text"
```

## WebSocket Protocol

Connect to `ws://localhost:8000/listen`:

1. Send JSON options (`uid`, `language`, `use_vad`, `beam_size`, ...)
2. Send audio as binary frames (16 kHz float32 PCM)
3. Send `b"END_OF_AUDIO"` to flush
4. Receive JSON segments: `{"uid": "...", "segments": [{"start", "end", "text", "completed"}]}`

## Tests

```bash
pytest tests/ -v
```
