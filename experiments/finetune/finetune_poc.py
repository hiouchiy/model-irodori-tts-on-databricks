"""LoRA fine-tuning feasibility PoC for Irodori-TTS on AI Runtime.

End-to-end, self-contained:
  1. Build the base InferenceRuntime and synthesize a handful of short clips
     (so we have coherent text<->audio pairs without an external dataset).
  2. Encode each clip to a DACVAE latent with the runtime's own codec and write
     a training manifest (text + latent_path + num_frames) — the exact format
     train.py / prepare_manifest.py expect.
  3. Run a SHORT LoRA training run via train.py (configs/train_v4_small_lora.yaml).
  4. Locate the saved PEFT adapter and synthesize once WITH the adapter applied,
     confirming the adapter loads and inference still works.

This answers "can we fine-tune this model on AIR?" — it is a pipeline smoke test,
not a quality run (8 clips / ~30 steps overfits by design).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
    save_wav,
)

BASE_REPO = "Aratako/Irodori-TTS-v4.1-Small"
WORK = Path("/tmp/ft")
LATENT_DIR = WORK / "latents"
MANIFEST = WORK / "manifest.jsonl"
LORA_OUT = WORK / "lora_out"
CONFIG = REPO_ROOT / "configs" / "train_v4_small_lora.yaml"

TEXTS = [
    "データブリックスは、データとAIの統合プラットフォームです。",
    "レイクハウスは、データレイクとデータウェアハウスを統合します。",
    "ユニティカタログは、データとAIの統合ガバナンスを提供します。",
    "モザイクAIで、エージェントを簡単に構築できます。",
    "デルタレイクは、信頼性の高いデータ基盤を実現します。",
    "スパークにより、大規模データを高速に処理できます。",
    "エムエルフローで、機械学習のライフサイクルを管理します。",
    "ジーニーを使えば、自然言語でデータに質問できます。",
]


def build_manifest(runtime: InferenceRuntime) -> None:
    LATENT_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, text in enumerate(TEXTS):
        res = runtime.synthesize(
            SamplingRequest(text=text, no_ref=True, seed=1000 + i, max_text_len=256),
            log_fn=None,
        )
        wav = res.audio.detach().to("cpu", dtype=torch.float32)  # (1, T) @ 48kHz
        with torch.inference_mode():
            latent = runtime.codec.encode_waveform(wav, sample_rate=res.sample_rate)[0].cpu()
        lp = LATENT_DIR / f"{i:04d}.pt"
        torch.save(latent, lp)
        entries.append(
            {
                "text": text,
                "latent_path": str(lp.relative_to(WORK)),
                "num_frames": int(latent.shape[0]),
            }
        )
        print(f"[ft] clip {i}: audio={wav.shape[-1] / res.sample_rate:.1f}s latent={tuple(latent.shape)}")
    MANIFEST.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n")
    print(f"[ft] wrote manifest with {len(entries)} entries -> {MANIFEST}")


def run_training(base_ckpt: str) -> None:
    cmd = [
        sys.executable, str(REPO_ROOT / "train.py"),
        "--config", str(CONFIG),
        "--manifest", str(MANIFEST),
        "--init-checkpoint", base_ckpt,
        "--output-dir", str(LORA_OUT),
        "--device", "cuda",
        "--precision", "bf16",
        "--max-steps", "30",
        "--batch-size", "2",
        "--num-workers", "0",
        "--warmup-steps", "3",
        "--stable-steps", "15",
        "--lr", "0.0005",
        "--save-every", "30",
        "--log-every", "5",
        "--valid-ratio", "0",
    ]
    print("[ft] launching training:\n  " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    print(f"[ft] training finished in {time.perf_counter() - t0:.1f}s")


def find_adapter() -> Path | None:
    cands = sorted(LORA_OUT.rglob("adapter_config.json"), key=lambda p: p.stat().st_mtime)
    return cands[-1].parent if cands else None


def main() -> None:
    try:
        sys.path.insert(0, str(REPO_ROOT / "air_job"))
        import airlog
        airlog.env_report()
    except Exception as _e:  # never let logging break the run
        print("[ft] (airlog unavailable)", _e)
    print(f"[ft] device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    base_ckpt = download_hf_checkpoint(BASE_REPO)

    runtime = InferenceRuntime.from_key(
        RuntimeKey(checkpoint=base_ckpt, model_device="cuda", model_precision="bf16",
                   codec_device="cuda", codec_precision="bf16")
    )
    build_manifest(runtime)
    del runtime
    torch.cuda.empty_cache()

    run_training(base_ckpt)

    adapter = find_adapter()
    out = WORK / "samples"
    out.mkdir(parents=True, exist_ok=True)
    if adapter is None:
        print("[ft] WARNING: no adapter_config.json found under output-dir; listing tree:")
        for p in sorted(LORA_OUT.rglob("*")):
            print("   ", p.relative_to(LORA_OUT))
        raise SystemExit("adapter not found")

    print(f"[ft] adapter dir: {adapter}")
    # Synthesize WITH the LoRA adapter applied at runtime.
    rt = InferenceRuntime.from_key(
        RuntimeKey(checkpoint=base_ckpt, model_device="cuda", model_precision="bf16",
                   codec_device="cuda", codec_precision="bf16")
    )
    text = "データブリックスで、ファインチューニングした音声を生成しています。"
    res = rt.synthesize(
        SamplingRequest(text=text, no_ref=True, seed=7, max_text_len=256, lora_adapter=str(adapter)),
        log_fn=None,
    )
    save_wav(out / "after_lora.wav", res.audio, res.sample_rate)
    print(f"[ft] LoRA inference OK: {res.audio.shape[-1] / res.sample_rate:.1f}s -> {out/'after_lora.wav'}")

    # Copy adapter (small) so it can be inspected/registered later.
    import shutil
    shutil.copytree(adapter, out / "adapter", dirs_exist_ok=True)
    print("[ft] DONE")


if __name__ == "__main__":
    main()
