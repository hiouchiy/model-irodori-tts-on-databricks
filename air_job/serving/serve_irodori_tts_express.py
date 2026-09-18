# Databricks notebook source
# MAGIC %md
# MAGIC # Irodori-TTS — GPU Express Deployment (env_pack)
# MAGIC
# MAGIC Register Irodori-TTS-v4.1-Small for Model Serving using **`env_pack="databricks_model_serving"`**,
# MAGIC which packs THIS serverless-GPU notebook environment (torch 2.10/cu128 + git deps) and restores it
# MAGIC at serving verbatim — avoiding the standard serving container rebuild that force-reinstalls
# MAGIC torch 2.7.1+cu118 (which broke the earlier standard-serving attempt: `libcudart.so.12 not found`).
# MAGIC
# MAGIC **Run on Serverless GPU (AI Runtime, `GPU_1xA10`, client image).** env_pack needs a serverless
# MAGIC client image (Classic fails `is_client_image=False`) and `mlflow==3.12.0`.
# MAGIC
# MAGIC **Dependency notes (learned the hard way — validated on CPU before this GPU run):**
# MAGIC - `dacvae` hard-imports `audiotools` (descript-audiotools 0.7.2), which declares a STALE
# MAGIC   `protobuf<3.20` pin. protobuf 3.19.6 lacks `google.protobuf.internal.builder` and BREAKS the
# MAGIC   serverless notebook kernel itself (pyspark/Spark-Connect needs protobuf>=3.20). It also
# MAGIC   conflicts with `databricks-sdk` (needs protobuf>=4.25.8).
# MAGIC - Fix = TWO-PHASE install: (1) install the real runtime deps with a safe `protobuf>=4.25.8,<6`;
# MAGIC   (2) install dacvae/silentcipher/descript-audiotools/argbind with `--no-deps` so their stale
# MAGIC   protobuf pin is ignored (audiotools only touches protobuf via tensorboard logging, never at
# MAGIC   import/inference — verified `import audiotools`/`dacvae` work on protobuf 5.29.4).
# MAGIC - The GPU client image preinstalls a `torchvision` built for its base torch (2.7.1); against
# MAGIC   our torch 2.10 it is ABI-broken (`partially initialized module 'torchvision'...`) when
# MAGIC   transformers probes it, and `%pip uninstall` can't remove the base-layer copy. Fix = install
# MAGIC   a MATCHING `torchvision` from the cu128 index (pip picks the torch-2.10 build).
# MAGIC - Do NOT install `databricks-sdk` explicitly (client image ships it; pyfunc never imports it).
# MAGIC - git deps break env_pack's dependency step ("Cannot find child https:") → register with
# MAGIC   `install_dependencies=False` (pack+restore, don't re-resolve).
# MAGIC - `databricks-sdk>=0.102.0` is REQUIRED: env_pack uploads a large `model_environment.tar`
# MAGIC   (torch/cu128 stack); older sdk hits a 5-min upload TimeoutError. (Now installable since
# MAGIC   descript-audiotools is `--no-deps`, so its protobuf<3.20 no longer clashes with sdk's >=4.25.8;
# MAGIC   pin `protobuf>=5.29.5,<6` to satisfy sdk's exclusions + keep pyspark alive.)
# MAGIC - We do NOT bundle the HF cache (it made the upload too big) — the pyfunc downloads the
# MAGIC   checkpoint/codec/tokenizer from the Hub at serving load (Express serving env has egress).

# COMMAND ----------

# Phase 1 — real runtime deps + a protobuf floor that keeps the notebook kernel (pyspark) alive.
# MAGIC %pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
# MAGIC   "torch==2.10.0" "torchvision" "torchaudio==2.10.0" "torchcodec==0.10.0" "torchao==0.16.0" \
# MAGIC   "transformers==5.12.1" "peft>=0.18.0" "safetensors>=0.7.0" "sentencepiece>=0.1.99,<0.2" \
# MAGIC   "soundfile>=0.12.0" "numba>=0.57.0" "llvmlite>=0.40.0" "huggingface-hub>=0.34.0" \
# MAGIC   "pyyaml>=6.0" "tqdm>=4.67.3" "torch-stoi>=0.2.3" "pydub>=0.25.1" "mlflow==3.12.0" \
# MAGIC   "databricks-sdk>=0.102.0" "einops>=0.8.0" "protobuf>=5.29.5,<6" \
# MAGIC   "ffmpy" "flatten-dict" "importlib-resources" "julius" "librosa" "markdown2" \
# MAGIC   "matplotlib" "pyloudnorm" "pystoi" "randomname" "rich" "scipy" "tensorboard" \
# MAGIC   "docstring-parser" "flask" "pyaml"

# COMMAND ----------

# Phase 2 — packages whose stale protobuf<3.20 pin must be ignored → --no-deps.
# MAGIC %pip install --no-deps \
# MAGIC   "git+https://github.com/facebookresearch/dacvae" \
# MAGIC   "git+https://github.com/SesameAILabs/silentcipher.git@d46d7d0893a583d8968ab3a6626e2289faec9152" \
# MAGIC   "descript-audiotools==0.7.2" "argbind==0.3.9"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# ---- inline logging helpers (self-diagnosing; the notebook print log is the record) ----
import time, sys, traceback, contextlib

def banner(msg):
    print(f"\n[log {time.strftime('%H:%M:%S')}] ===== {msg} =====", flush=True)

@contextlib.contextmanager
def stage(name):
    banner(f"BEGIN {name}"); t0 = time.perf_counter()
    try:
        yield
    except BaseException:
        print(f"[log] EXCEPTION in {name} after {time.perf_counter()-t0:.1f}s:", flush=True)
        traceback.print_exc(); raise
    else:
        print(f"[log] OK {name} ({time.perf_counter()-t0:.1f}s)", flush=True)

def env_report():
    banner("ENV REPORT")
    print("python:", sys.version.split()[0], flush=True)
    from importlib.metadata import version, PackageNotFoundError
    import torch
    print("torch:", torch.__version__, "| cuda_avail:", torch.cuda.is_available(),
          "| torch.version.cuda:", torch.version.cuda, flush=True)
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        print("gpu:", torch.cuda.get_device_name(0), f"cc={cc[0]}.{cc[1]}", flush=True)
    for p in ["torch", "torchaudio", "torchcodec", "torchao", "transformers", "mlflow",
              "protobuf", "descript-audiotools", "dacvae", "silentcipher", "numba", "soundfile"]:
        try:
            print(f"  {p}: {version(p)}", flush=True)
        except PackageNotFoundError:
            print(f"  {p}: (not installed)", flush=True)

env_report()

# COMMAND ----------

# Serverless notebooks: work on local disk (WSFS is slow/limited for big files).
import os, subprocess, tempfile, json
os.chdir(tempfile.mkdtemp())

REPO_DIR = "/tmp/irodori"
HF_CACHE = "/tmp/hfcache"
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

with stage("git clone Irodori-TTS"):
    subprocess.run(["rm", "-rf", REPO_DIR], check=False)
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/Aratako/Irodori-TTS.git", REPO_DIR], check=True)
    sys.path.insert(0, REPO_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Populate the HF cache + smoke-test on this GPU (cache is bundled → served offline)

# COMMAND ----------

import torch
from irodori_tts.inference_runtime import (
    InferenceRuntime, RuntimeKey, SamplingRequest, download_hf_checkpoint, save_wav,
)

REPO_ID = "Aratako/Irodori-TTS-v4.1-Small"
with stage("build runtime + smoke synth (populates HF cache)"):
    ckpt = download_hf_checkpoint(REPO_ID)
    rt = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=ckpt, model_device="cuda", model_precision="bf16",
        codec_device="cuda", codec_precision="bf16"))
    res = rt.synthesize(SamplingRequest(
        text="こんにちは。これはデータブリックスのモデルサービング上で動く音声合成のテストです。",
        no_ref=True, max_text_len=512, seed=42), log_fn=None)
    save_wav("/tmp/express_smoke.wav", res.audio, res.sample_rate)
    print("smoke ok:", round(res.audio.shape[-1] / res.sample_rate, 2), "s", flush=True)
    del rt; torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write the pyfunc (offline: reads bundled HF cache; no Hub egress at serving)

# COMMAND ----------

PYFUNC_SRC = r'''
"""Irodori-TTS serving pyfunc — general-purpose, agent-friendly API.

Request (MLflow dataframe_records): each row = {"text": <str>, "params_json": <str>}.
`params_json` is a JSON object accepting ANY field of the model's SamplingRequest
(discovered dynamically via dataclasses.fields, so it auto-tracks the model version),
plus `ref_wav_b64` (base64 WAV for voice cloning). A single opaque JSON string is used
(rather than typed columns) so MLflow schema enforcement never drops unknown tuning knobs.

Introspection: send params_json '{"__list_params__": true}' to get the supported
parameter names + defaults back (no synthesis) — lets an agent self-discover the surface.

Long text: this endpoint synthesizes ONE utterance per row (model max_text_len ~256 tokens,
duration cap ~30s). For long passages, the caller splits into sentences and sends multiple
rows, then concatenates the returned clips (the companion app does exactly this).

Response row: audio_wav_b64, sample_rate, audio_seconds, synth_seconds, used_seed,
messages (runtime notes e.g. duration clamping), applied_params, error.
"""
import base64, io, os, time, json, tempfile, dataclasses
import mlflow.pyfunc
import pandas as pd

_INT_FIELDS = {"num_steps", "seed", "num_candidates", "max_text_len", "max_caption_len",
               "speaker_kv_max_layers", "tail_window_size"}
_REF_FIELDS = ("ref_wav", "ref_wavs", "ref_latent", "ref_latents", "ref_embed")


class IrodoriTTSModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch
        from irodori_tts.inference_runtime import (
            InferenceRuntime, RuntimeKey, download_hf_checkpoint, SamplingRequest,
        )
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        cfg = context.model_config or {}
        repo = cfg.get("hf_checkpoint", "Aratako/Irodori-TTS-v4.1-Small")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        precision = cfg.get("precision", "bf16" if device == "cuda" else "fp32")
        ckpt = download_hf_checkpoint(repo)
        self._runtime = InferenceRuntime.from_key(RuntimeKey(
            checkpoint=ckpt, model_device=device, model_precision=precision,
            codec_device=device, codec_precision=precision))
        self._SamplingRequest = SamplingRequest
        # The full tunable surface = every field the model's SamplingRequest defines.
        self._fields = {f.name for f in dataclasses.fields(SamplingRequest)}
        self._defaults = {
            f.name: (None if f.default is dataclasses.MISSING else f.default)
            for f in dataclasses.fields(SamplingRequest) if f.name != "text"
        }

    def _synth(self, text, params):
        import soundfile as sf, torch
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

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            rows = model_input.to_dict(orient="records")
        elif isinstance(model_input, dict):
            rows = [model_input]
        else:
            rows = list(model_input)

        out = []
        for row in rows:
            pj = row.get("params_json")
            if isinstance(pj, str) and pj.strip():
                p = json.loads(pj)
            elif isinstance(pj, dict):
                p = pj
            else:
                p = {}
            if p.get("__list_params__"):  # agent self-discovery, no synthesis
                out.append({"audio_wav_b64": "", "sample_rate": 0, "audio_seconds": 0.0,
                            "synth_seconds": 0.0, "used_seed": 0,
                            "messages": json.dumps({"supported_params": sorted(self._fields),
                                                    "extras": ["ref_wav_b64"],
                                                    "defaults": self._defaults}, ensure_ascii=False, default=str),
                            "applied_params": "{}", "error": ""})
                continue
            out.append(self._synth(row.get("text", ""), p))
        return pd.DataFrame(out)


mlflow.models.set_model(IrodoriTTSModel())
'''
with open("/tmp/tts_pyfunc_express.py", "w") as f:
    f.write(PYFUNC_SRC)
print("wrote pyfunc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log (no registered_model_name) then register with env_pack (install_dependencies=False)

# COMMAND ----------

import mlflow, pandas as pd
mlflow.set_registry_uri("databricks-uc")
# Two stable string columns: `text` + `params_json` (opaque JSON of any SamplingRequest field).
input_example = pd.DataFrame([{
    "text": "こんにちは、これはテストです。",
    "params_json": json.dumps({"num_steps": 40, "cfg_scale_text": 3.0}),
}])
# UC model to register as: catalog.schema.name. Override via the notebook widget or the
# job's base_parameters (key "uc_model"); the default uses main.default which most workspaces
# have. You need CREATE MODEL on the target schema. See docs/GUIDE.md.
dbutils.widgets.text("uc_model", "main.default.irodori_tts_express", "UC model (catalog.schema.name)")
UC_MODEL = dbutils.widgets.get("uc_model")

with stage("mlflow.pyfunc.log_model"):
    info = mlflow.pyfunc.log_model(
        name="irodori_tts",
        python_model="/tmp/tts_pyfunc_express.py",
        code_paths=[os.path.join(REPO_DIR, "irodori_tts")],
        model_config={"hf_checkpoint": REPO_ID, "precision": "bf16"},
        input_example=input_example,
        extra_pip_requirements=["mlflow==3.12.0"],  # no URLs/git → keeps env_pack happy
    )
    print("model_uri:", info.model_uri, flush=True)

with stage("mlflow.register_model(env_pack, install_dependencies=False)"):
    from mlflow.utils.env_pack import EnvPackConfig
    mv = mlflow.register_model(
        model_uri=info.model_uri, name=UC_MODEL,
        env_pack=EnvPackConfig(name="databricks_model_serving", install_dependencies=False))
    print("registered:", UC_MODEL, "version", mv.version, flush=True)

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"uc_model": UC_MODEL, "version": mv.version}))
