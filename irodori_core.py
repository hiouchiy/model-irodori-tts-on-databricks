"""Irodori-TTS core, shared by the notebook, the AI Runtime batch job and the serving model.

One place for: the runtime pins, the pinned upstream commit, the checkpoint, preparing a working directory (clone),
and the request contract (`text` + `params_json` -> one WAV; docs/SERVING-API.md). The three ways to run
(notebooks/generate.py, air_job/generate.yaml via generate.py, serving/irodori_pyfunc.py) only add their own input
and output handling on top.

Importing this module needs neither torch nor Irodori-TTS; they are imported when a Synthesizer is built.
"""
from __future__ import annotations

import base64
import dataclasses
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Serverless GPU (notebooks and AI Runtime) presets HF_HUB_ENABLE_HF_TRANSFER=1 without installing hf_transfer, and
# every Hub download then raises. huggingface_hub reads it at import, so switch it off before anything imports it.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

UPSTREAM_REPO = "https://github.com/Aratako/Irodori-TTS.git"
UPSTREAM_SHA = "89f9d8fbd4d51ea019867ee1197725ede1df13c5"   # 2026-09-12 (pinned)
HF_CHECKPOINT = "Aratako/Irodori-TTS-v4.1-Small"

# The runtime: torch 2.10 / cu128, as upstream's uv.lock. scripts/install.sh installs it in two phases.
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
# Phase 1: the real runtime deps, with a protobuf floor that keeps the serverless notebook kernel (pyspark) alive.
# - torchvision is pinned to the build for torch 2.10: the GPU image ships one for its base torch 2.7.1 that is
#   ABI-broken against 2.10 and cannot be uninstalled from the base layer.
# - databricks-sdk>=0.102.0: env_pack uploads a large environment tar; older sdks time out after 5 min.
RUNTIME_REQUIREMENTS = [
    "torch==2.10.0", "torchvision==0.25.0", "torchaudio==2.10.0", "torchcodec==0.10.0", "torchao==0.16.0",
    "transformers==5.12.1", "peft>=0.18.0", "safetensors>=0.7.0", "sentencepiece>=0.1.99,<0.2",
    "soundfile>=0.12.0", "numba>=0.57.0", "llvmlite>=0.40.0", "huggingface-hub>=0.34.0",
    "pyyaml>=6.0", "tqdm>=4.67.3", "torch-stoi>=0.2.3", "pydub>=0.25.1", "mlflow==3.12.0",
    "databricks-sdk>=0.102.0", "einops>=0.8.0", "protobuf>=5.29.5,<6",
    "ffmpy", "flatten-dict", "importlib-resources", "julius", "librosa", "markdown2",
    "matplotlib", "pyloudnorm", "pystoi", "randomname", "rich", "scipy", "tensorboard",
    "docstring-parser", "flask", "pyaml",
]
# Phase 2, with --no-deps: dacvae imports audiotools (descript-audiotools 0.7.2), which pins protobuf<3.20. That
# breaks the notebook kernel and clashes with databricks-sdk; audiotools only uses protobuf for tensorboard logging.
NO_DEPS_REQUIREMENTS = [
    "git+https://github.com/facebookresearch/dacvae",
    "git+https://github.com/SesameAILabs/silentcipher.git@d46d7d0893a583d8968ab3a6626e2289faec9152",
    "descript-audiotools==0.7.2", "argbind==0.3.9",
]
# The serving model's own pip requirement; env_pack restores the rest of the registering environment as is.
SERVING_PIP_REQUIREMENTS = ["mlflow==3.12.0"]

_INT_FIELDS = {"num_steps", "seed", "num_candidates", "max_text_len", "max_caption_len",
               "speaker_kv_max_layers", "tail_window_size"}
_REF_FIELDS = ("ref_wav", "ref_wavs", "ref_latent", "ref_latents", "ref_embed")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare(work: str | Path = "/tmp/irodori") -> Path:
    """Clone Irodori-TTS at the pinned commit into `work` and make `irodori_tts` importable. Returns the clone."""
    repo = Path(work) / "Irodori-TTS"
    if not repo.is_dir():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", UPSTREAM_REPO, str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", UPSTREAM_SHA], check=True)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def parse_params(params_json: Any) -> dict:
    """The `params_json` column: a JSON object string, a dict, or empty."""
    if isinstance(params_json, str) and params_json.strip():
        return json.loads(params_json)
    if isinstance(params_json, dict):
        return params_json
    return {}


class Synthesizer:
    """One loaded Irodori-TTS runtime. `synthesize` returns one response row of the endpoint."""

    def __init__(self, checkpoint: str = HF_CHECKPOINT, precision: str | None = None):
        import torch
        from irodori_tts.inference_runtime import (InferenceRuntime, RuntimeKey, SamplingRequest,
                                                   download_hf_checkpoint)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        precision = precision or ("bf16" if device == "cuda" else "fp32")
        ckpt = download_hf_checkpoint(checkpoint)
        self._runtime = InferenceRuntime.from_key(RuntimeKey(
            checkpoint=ckpt, model_device=device, model_precision=precision,
            codec_device=device, codec_precision=precision))
        self._SamplingRequest = SamplingRequest
        # The full tunable surface = every field the model's SamplingRequest defines.
        self._fields = {f.name for f in dataclasses.fields(SamplingRequest)}
        self._defaults = {f.name: (None if f.default is dataclasses.MISSING else f.default)
                          for f in dataclasses.fields(SamplingRequest) if f.name != "text"}

    def list_params(self) -> dict:
        """The `{"__list_params__": true}` response row (no synthesis)."""
        return {"audio_wav_b64": "", "sample_rate": 0, "audio_seconds": 0.0, "synth_seconds": 0.0, "used_seed": 0,
                "messages": json.dumps({"supported_params": sorted(self._fields), "extras": ["ref_wav_b64"],
                                        "defaults": self._defaults}, ensure_ascii=False, default=str),
                "applied_params": "{}", "error": ""}

    def synthesize(self, text: str, params: dict) -> dict:
        """One utterance. Errors come back in the row's `error`, not as exceptions."""
        import soundfile as sf
        import torch
        kwargs, tmp = {}, None

        ref_b64 = params.get("ref_wav_b64")
        if ref_b64:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(base64.b64decode(ref_b64)); tmp.flush(); tmp.close()
            kwargs["ref_wav"] = tmp.name

        for k, v in params.items():
            if k in self._fields and k != "text" and v is not None:
                kwargs[k] = int(v) if k in _INT_FIELDS else v

        # Default to speaker-free synthesis unless the caller supplied a reference.
        if not any(k in kwargs for k in _REF_FIELDS) and "no_ref" not in kwargs:
            kwargs["no_ref"] = True

        try:
            t0 = time.perf_counter()
            r = self._runtime.synthesize(self._SamplingRequest(text=str(text), **kwargs), log_fn=None)
            dt = time.perf_counter() - t0
            a = r.audio.detach().to("cpu", dtype=torch.float32)
            a = a.squeeze(0).numpy() if a.shape[0] == 1 else a.T.numpy()
            # In memory: libsndfile seeks while writing WAV headers, which UC Volume (FUSE) paths reject.
            buf = io.BytesIO(); sf.write(buf, a, r.sample_rate, format="WAV")
            applied = {k: ("<ref_wav>" if k == "ref_wav" else v) for k, v in kwargs.items()}
            return {"audio_wav_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
                    "sample_rate": int(r.sample_rate),
                    "audio_seconds": round(r.audio.shape[-1] / r.sample_rate, 3),
                    "synth_seconds": round(dt, 3),
                    "used_seed": int(r.used_seed),
                    "messages": json.dumps(list(r.messages or []), ensure_ascii=False),
                    "applied_params": json.dumps(applied, ensure_ascii=False),
                    "error": ""}
        except Exception as e:
            return {"audio_wav_b64": "", "sample_rate": 0, "audio_seconds": 0.0,
                    "synth_seconds": 0.0, "used_seed": 0, "messages": "[]",
                    "applied_params": json.dumps({k: str(v) for k, v in kwargs.items()}),
                    "error": f"{type(e).__name__}: {e}"}
        finally:
            if tmp:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass

    def handle(self, text: str, params_json: Any) -> dict:
        """One request row: `text` + `params_json` -> one response row."""
        p = parse_params(params_json)
        if p.get("__list_params__"):  # agent self-discovery, no synthesis
            return self.list_params()
        return self.synthesize(text, p)
