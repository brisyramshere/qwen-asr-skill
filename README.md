# qwen-asr-skill

A [Claude Code](https://claude.ai/code) skill for local speech-to-text transcription using [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B).

## Features

- **52 languages** and 22 Chinese dialects
- **Multiple audio formats**: WAV, MP3, FLAC, M4A, OGG, etc.
- **Auto GPU/CPU fallback**: tries GPU (float16) first, falls back to CPU (float32) if VRAM is insufficient
- **Long audio support**: automatically splits audio at silence boundaries using [silero-vad](https://github.com/snakers4/silero-vad), transcribes chunk by chunk, then concatenates results
- **JSON output**: structured output for easy integration

## Setup

### 1. Create virtual environment

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install qwen-asr soundfile silero-vad
```

For GPUs with compute capability < 7.0 (e.g. GTX 1060 Pascal), install PyTorch 2.4.x with CUDA 11.8:

```bash
pip install torch==2.4.1 torchaudio==2.4.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

For modern GPUs (RTX 20xx+), the default PyTorch from PyPI works fine.

### 3. Download model (optional)

The model (~4.7GB) downloads automatically on first run. To pre-download:

```bash
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ~/models/Qwen3-ASR-1.7B
```

## Usage

```bash
python qwen-asr-skill/scripts/transcribe.py <audio_path> [options]
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `audio_path` | Path to audio file (required) | - |
| `--language` | Force language (e.g. Chinese, English). Auto-detect if omitted | Auto-detect |
| `--device` | Inference device: `auto` / `cuda` / `cpu` | `auto` |
| `--model-path` | Model path or HuggingFace ID | `~/models/Qwen3-ASR-1.7B` |
| `--max-chunk-sec` | Max chunk duration (seconds) for VAD splitting | `90` |
| `--max-new-tokens` | Max tokens to generate. Increase for long audio | `2048` |

### Examples

```bash
# Basic transcription
python qwen-asr-skill/scripts/transcribe.py recording.wav

# Force language
python qwen-asr-skill/scripts/transcribe.py podcast.mp3 --language English

# CPU-only inference
python qwen-asr-skill/scripts/transcribe.py lecture.flac --device cpu
```

### Output

JSON to stdout, status info to stderr:

```json
{"language": "Chinese", "text": "转写出的文字内容"}
```

## How long audio works

Audio longer than `--max-chunk-sec` (default 90s) is automatically processed as:

1. **VAD detection** — silero-vad finds silence boundaries in the audio
2. **Smart splitting** — audio is split at silence points (never mid-sentence)
3. **Chunk transcription** — each chunk is transcribed independently
4. **Result concatenation** — all chunks are joined into the final text

This prevents GPU OOM errors on limited VRAM hardware while preserving sentence integrity.

## Install as Claude Code skill

Copy the `qwen-asr-skill/` directory to your Claude Code skills folder:

```bash
cp -r qwen-asr-skill ~/.claude/skills/asr
```

## License

MIT
