#!/usr/bin/env python3
"""Quantization performance sweep for Irodori-TTS on AI Runtime.

For each precision / quantization variant we build the InferenceRuntime once,
run a warmup synthesis, then time N synthesis calls with a fixed text/seed and
identical sampling settings (only the model weights differ). We report:

  - load_s          : wall time to build the runtime (download excluded on warm cache)
  - weights_MiB     : GPU memory attributable to model weights after load
  - peak_infer_MiB  : peak GPU memory during a synthesis call
  - model_s (mean)  : SamplingResult.total_to_decode (tokenize -> latent decode)
  - wall_s (mean)   : end-to-end synthesize() wall time
  - audio_s         : duration of generated audio
  - RTF             : audio_s / wall_s  (>1 = faster than real time)

Everything else (num_steps=40, seed, codec on CUDA/bf16, text) is held constant
so the numbers isolate the effect of quantizing the RF-DiT backbone.
"""
from __future__ import annotations

import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

# Script lives in <repo>/air_job/; make the repo root importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
    save_wav,
)

BASE_REPO = "Aratako/Irodori-TTS-v4.1-Small"
QUANT_REPO = "Aratako/Irodori-TTS-v4.1-Small-Quantized"

# (label, hf source, model_precision). Quantized checkpoints are stored bf16.
VARIANTS = [
    ("fp32", BASE_REPO, "fp32"),
    ("bf16", BASE_REPO, "bf16"),
    ("int8-weight-only", f"{QUANT_REPO}/int8-weight-only", "bf16"),
    ("int8-dynamic", f"{QUANT_REPO}/int8-dynamic", "bf16"),
    ("int4-weight-only", f"{QUANT_REPO}/int4-weight-only", "bf16"),
    # float8 requires compute capability >= 8.9 (Ada/Hopper); skipped on A10 (8.6).
    ("float8-weight-only", f"{QUANT_REPO}/float8-weight-only", "bf16"),
    ("float8-dynamic", f"{QUANT_REPO}/float8-dynamic", "bf16"),
]

N_TIMED = 5
TEXT = (
    "こんにちは。今日はデータブリックスについてご紹介します。"
    "データブリックスは、データとAIを一つに統合した、"
    "データインテリジェンスプラットフォームです。"
    "レイクハウスアーキテクチャを基盤に、データエンジニアリング、分析、"
    "機械学習、そして生成AIまでを、単一の基盤で扱うことができます。"
)
SEED = 12345
OUT_DIR = Path("/tmp/tts_bench")


def gpu_mib() -> float:
    return torch.cuda.memory_allocated() / (1024**2)


def bench_one(label: str, source: str, precision: str) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = gpu_mib()

    t0 = time.perf_counter()
    ckpt = download_hf_checkpoint(source)
    ckpt_mib = Path(ckpt).stat().st_size / (1024**2)
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=ckpt,
            model_device="cuda",
            model_precision=precision,
            codec_device="cuda",
            codec_precision="bf16",
        )
    )
    load_s = time.perf_counter() - t0
    weights_mib = gpu_mib() - mem_before

    def synth():
        return runtime.synthesize(
            SamplingRequest(
                text=TEXT, no_ref=True, seed=SEED, max_text_len=512, max_seconds=30.0
            ),
            log_fn=None,
        )

    synth()  # warmup (kernels/autotune)

    torch.cuda.reset_peak_memory_stats()
    model_times, wall_times = [], []
    result = None
    for _ in range(N_TIMED):
        torch.cuda.synchronize()
        w0 = time.perf_counter()
        result = synth()
        torch.cuda.synchronize()
        wall_times.append(time.perf_counter() - w0)
        model_times.append(result.total_to_decode)
    peak_infer_mib = torch.cuda.max_memory_allocated() / (1024**2)

    audio_s = result.audio.shape[-1] / result.sample_rate
    wall_mean = statistics.mean(wall_times)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_wav(OUT_DIR / f"{label}.wav", result.audio, result.sample_rate)

    row = {
        "variant": label,
        "precision": precision,
        "ckpt_MiB": round(ckpt_mib, 1),
        "load_s": round(load_s, 2),
        "weights_MiB": round(weights_mib, 1),
        "peak_infer_MiB": round(peak_infer_mib, 1),
        "model_s_mean": round(statistics.mean(model_times), 3),
        "wall_s_mean": round(wall_mean, 3),
        "wall_s_std": round(statistics.pstdev(wall_times), 3),
        "audio_s": round(audio_s, 2),
        "RTF": round(audio_s / wall_mean, 2),
    }

    del runtime, result
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main() -> None:
    try:
        import airlog  # air_job/ is this script's own dir → importable directly
        airlog.env_report()
    except Exception as _e:  # never let logging break the run
        print("[bench] (airlog unavailable)", _e)
    dev = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"[bench] device={dev} compute_capability={cc[0]}.{cc[1]} torch={torch.__version__}")
    rows = []
    for label, source, precision in VARIANTS:
        print(f"\n[bench] ===== {label} ({precision}) =====", flush=True)
        try:
            row = bench_one(label, source, precision)
            rows.append(row)
            print(f"[bench] {json.dumps(row, ensure_ascii=False)}", flush=True)
        except Exception as exc:  # keep going; some variants need newer HW
            print(f"[bench] SKIP {label}: {type(exc).__name__}: {exc}", flush=True)
            rows.append({"variant": label, "precision": precision, "error": f"{type(exc).__name__}: {exc}"})

    # Pretty table
    print("\n[bench] ===== SUMMARY =====")
    cols = ["variant", "ckpt_MiB", "weights_MiB", "peak_infer_MiB", "model_s_mean", "wall_s_mean", "audio_s", "RTF"]
    print(" | ".join(f"{c:>14}" for c in cols))
    for r in rows:
        if "error" in r:
            print(f"{r['variant']:>14} | ERROR: {r['error']}")
        else:
            print(" | ".join(f"{str(r.get(c, '')):>14}" for c in cols))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\n[bench] wrote {OUT_DIR/'results.json'}")


if __name__ == "__main__":
    sys.exit(main())
