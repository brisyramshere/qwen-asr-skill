#!/usr/bin/env python3
"""
Qwen3-ASR-1.7B 本地语音转文本推理脚本

使用方法:
    python transcribe.py <audio_path> [options]

示例:
    python transcribe.py /path/to/audio.wav
    python transcribe.py /path/to/audio.mp3 --language Chinese
    python transcribe.py /path/to/audio.flac --device cpu
"""

import argparse
import json
import os
import sys
import tempfile
import time

DEFAULT_MODEL = os.environ.get(
    "QWEN_ASR_MODEL_PATH",
    os.path.expanduser("~/models/Qwen3-ASR-1.7B"),
)
FALLBACK_MODEL = "Qwen/Qwen3-ASR-1.7B"


def get_model_path(args_model_path: str) -> str:
    """Resolve model path: CLI arg > env var > local dir > HuggingFace ID."""
    if args_model_path:
        return args_model_path
    if os.path.isdir(DEFAULT_MODEL):
        return DEFAULT_MODEL
    return FALLBACK_MODEL


def _gpu_compatible() -> bool:
    """Check if GPU supports the current PyTorch CUDA build."""
    import torch
    if not torch.cuda.is_available():
        return False
    try:
        # Test actual CUDA operation (catches sm_xx incompatibility)
        torch.zeros(1, device="cuda:0")
        return True
    except Exception:
        return False


def load_model(model_path: str, device: str, max_new_tokens: int):
    """Load Qwen3-ASR model with GPU/CPU fallback."""
    import torch
    from qwen_asr import Qwen3ASRModel

    if device == "auto":
        if _gpu_compatible():
            try:
                model = Qwen3ASRModel.from_pretrained(
                    model_path,
                    dtype=torch.float16,
                    device_map="cuda:0",
                    max_new_tokens=max_new_tokens,
                )
                print("INFO: Model loaded on GPU (float16)", file=sys.stderr)
                return model
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                print(f"WARN: GPU load failed ({e}), falling back to CPU", file=sys.stderr)
        else:
            print("INFO: GPU not compatible or not available, using CPU", file=sys.stderr)

        model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.float32,
            device_map="cpu",
            max_new_tokens=max_new_tokens,
        )
        print("INFO: Model loaded on CPU (float32)", file=sys.stderr)
        return model

    elif device == "cuda":
        model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="cuda:0",
            max_new_tokens=max_new_tokens,
        )
        print("INFO: Model loaded on GPU (float16)", file=sys.stderr)
        return model

    else:  # cpu
        model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.float32,
            device_map="cpu",
            max_new_tokens=max_new_tokens,
        )
        print("INFO: Model loaded on CPU (float32)", file=sys.stderr)
        return model


def vad_split(audio_path: str, max_chunk_sec: float = 90) -> list[str]:
    """Split long audio at silence boundaries using VAD. Returns list of file paths."""
    import torch
    import torchaudio
    from silero_vad import load_silero_vad, get_speech_timestamps
    import soundfile as sf

    wav, sr = torchaudio.load(audio_path)

    # Convert to 16kHz mono
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav.squeeze(0)

    duration = wav.shape[0] / sr
    if duration <= max_chunk_sec:
        return [audio_path]

    print(f"INFO: Audio is {duration:.1f}s, splitting with VAD (max {max_chunk_sec}s per chunk)", file=sys.stderr)

    vad_model = load_silero_vad()
    speech_ts = get_speech_timestamps(
        wav, vad_model,
        sampling_rate=16000,
        min_silence_duration_ms=300,
        speech_pad_ms=200,
        return_seconds=False,
    )

    if not speech_ts:
        print("WARN: VAD detected no speech, falling back to whole file", file=sys.stderr)
        return [audio_path]

    # Merge adjacent segments up to max_chunk_sec
    chunks = []
    current_start = speech_ts[0]['start']
    current_end = speech_ts[0]['end']

    for seg in speech_ts[1:]:
        merged_duration = (seg['end'] - current_start) / sr
        if merged_duration > max_chunk_sec:
            chunks.append((current_start, current_end))
            current_start = seg['start']
        current_end = seg['end']
    chunks.append((current_start, current_end))

    print(f"INFO: Split into {len(chunks)} chunks", file=sys.stderr)

    # Save each chunk as a temp wav file
    tmp_paths = []
    for i, (start, end) in enumerate(chunks):
        chunk_wav = wav[start:end].numpy()
        chunk_duration = (end - start) / sr
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(tmp.name, chunk_wav, 16000)
        tmp_paths.append(tmp.name)
        print(f"INFO: Chunk {i+1}/{len(chunks)}: {chunk_duration:.1f}s", file=sys.stderr)

    return tmp_paths


def transcribe(model, audio_path: str, language: str | None, max_chunk_sec: float) -> dict:
    """Run transcription, with VAD chunking for long audio."""
    chunks = vad_split(audio_path, max_chunk_sec)

    texts = []
    detected_lang = None
    for i, chunk_path in enumerate(chunks):
        if len(chunks) > 1:
            print(f"INFO: Transcribing chunk {i+1}/{len(chunks)}...", file=sys.stderr)
        results = model.transcribe(audio=chunk_path, language=language)
        r = results[0]
        texts.append(r.text)
        if detected_lang is None:
            detected_lang = r.language

    # Clean up temp files
    for p in chunks:
        if p != audio_path:
            try:
                os.unlink(p)
            except OSError:
                pass

    return {
        "language": detected_lang,
        "text": "".join(texts),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR-1.7B local speech-to-text transcription",
    )
    parser.add_argument(
        "audio_path",
        help="Path to audio file (wav, mp3, flac, m4a, ogg, etc.)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force language (e.g. Chinese, English). Auto-detect if omitted.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=f"Model path or HuggingFace ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use (default: auto = try GPU, fallback CPU)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max tokens to generate. Increase for very long audio (default: 2048)",
    )
    parser.add_argument(
        "--max-chunk-sec",
        type=float,
        default=90,
        help="Max chunk duration in seconds for VAD splitting. Long audio is split at silence boundaries (default: 90)",
    )

    args = parser.parse_args()

    # Validate audio file
    if not os.path.isfile(args.audio_path):
        print(json.dumps({"error": f"File not found: {args.audio_path}"}))
        sys.exit(1)

    model_path = get_model_path(args.model_path)

    t0 = time.time()
    model = load_model(model_path, args.device, args.max_new_tokens)
    load_time = time.time() - t0
    print(f"INFO: Model loaded in {load_time:.1f}s", file=sys.stderr)

    t1 = time.time()
    result = transcribe(model, args.audio_path, args.language, args.max_chunk_sec)
    infer_time = time.time() - t1
    print(f"INFO: Transcription completed in {infer_time:.1f}s", file=sys.stderr)

    # JSON output to stdout (for Claude to parse)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
