---
name: qwen-asr-skill
description: Local speech-to-text transcription using Qwen3-ASR-1.7B. Use this skill when a user provides an audio file path and needs speech recognition, transcription, or ASR. Supports 52 languages and dialects, and common audio formats including WAV, MP3, FLAC, M4A, and OGG.
---

# Qwen ASR Skill - Local Speech-to-Text

Transcribe audio files to text locally using the Qwen3-ASR-1.7B model.

## Setup

Install dependencies into a Python 3.10+ virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install qwen-asr soundfile silero-vad
```

For GPUs with compute capability < 7.0 (e.g. GTX 1060), install PyTorch 2.4.x with CUDA 11.8:

```bash
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

## Usage

Run the transcription script with the path to an audio file:

```bash
python scripts/transcribe.py <audio_path>
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `audio_path` | Absolute path to audio file (required) | - |
| `--language` | Force language (e.g. Chinese, English). Auto-detect if omitted | Auto-detect |
| `--device` | Inference device: auto / cuda / cpu | auto |
| `--model-path` | Model path or HuggingFace ID | ~/models/Qwen3-ASR-1.7B |
| `--max-chunk-sec` | Max chunk duration for VAD splitting. Long audio is split at silence boundaries | 90 |
| `--max-new-tokens` | Max tokens to generate. Increase for long audio | 2048 |

### Examples

Basic transcription:
```bash
python scripts/transcribe.py /path/to/audio.wav
```

Force language:
```bash
python scripts/transcribe.py /path/to/audio.mp3 --language Chinese
```

Force CPU inference:
```bash
python scripts/transcribe.py /path/to/audio.flac --device cpu
```

## Output Format

The script outputs JSON to stdout and status info to stderr:

```json
{"language": "Chinese", "text": "Transcribed text content"}
```

On error:
```json
{"error": "Error description"}
```

## Notes

- First run downloads the model (~4.7GB), cached for subsequent runs
- Auto mode: tries GPU (float16) first, falls back to CPU (float32) if VRAM is insufficient
- Supports: WAV, MP3, FLAC, M4A, OGG and other common audio formats
- 52 languages including Chinese, English, Japanese, Korean, French, German, etc.
- 22 Chinese dialects supported
- **Long audio**: Audio longer than 90s is automatically split at silence boundaries using silero-vad, transcribed chunk by chunk, then concatenated. This prevents OOM on limited VRAM GPUs.
