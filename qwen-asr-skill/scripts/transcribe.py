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


def transcribe(model, audio_path: str, language: str | None) -> dict:
    """Run transcription and return result dict."""
    lang_arg = language if language else None

    results = model.transcribe(
        audio=audio_path,
        language=lang_arg,
    )

    r = results[0]
    return {
        "language": r.language,
        "text": r.text,
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
    result = transcribe(model, args.audio_path, args.language)
    infer_time = time.time() - t1
    print(f"INFO: Transcription completed in {infer_time:.1f}s", file=sys.stderr)

    # JSON output to stdout (for Claude to parse)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
