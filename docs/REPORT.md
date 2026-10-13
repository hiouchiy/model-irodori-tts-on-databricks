# Irodori-TTS on Databricks AI Runtime — PoC report

**Model:** [Aratako/Irodori-TTS-v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) (0.8B; flow matching
RF-DiT + ModernBERT text encoder + DACVAE 48 kHz codec; Japanese only)
**Upstream:** https://github.com/Aratako/Irodori-TTS
**Environment:** Databricks AI Runtime (AIR CLI, Beta), serverless GPU, Model Serving
**Date:** 2026-09-16

The layout of this repo has changed since (one core, three ways to run; see the [README](../README.md)). The
findings below still hold, and the pins in `irodori_core.py` come from them.

---

## 0. Summary

| Phase | Result | In one line |
|---|---|---|
| 1. Inference on AIR | ✅ | A 27 s Databricks introduction on an A10. RF sampling 2.5 s (cold). |
| 2. Model Serving | ✅ with **env_pack (GPU Express Deployment)** | Standard GPU serving fails on a torch conflict; env_pack solves it, and the endpoint returns audio. |
| 3. LoRA fine-tuning | ✅ | 30 steps in 38.8 s; the adapter was saved and used for inference. |
| 4. Quantization | ✅ (A10 + H100) | **Quantization does not speed this model up. It only saves VRAM.** |

**The main technical point:** the model needs `torch>=2.10 / cu128`, `transformers 5.x` and git dependencies (dacvae,
silentcipher). AIR's `environment.dependencies` cannot reproduce this: environment 4 ships torch 2.7.1.

What worked for the PoC jobs:
- run `uv sync` inside the AIR job's `command:`;
- this reproduces the author's environment from the repo's `uv.lock`;
- it bypasses AIR's dependency resolution, so the pattern carries over to other bleeding-edge OSS models.

The repo's current ways to run install the same stack with pip in two phases (`scripts/install.sh`), so the notebook,
the AIR job and the env_pack registration share one recipe.

---

## 1. Inference on AIR ✅

- **Method:**
  1. snapshot the upstream repo as `code_source`;
  2. in `command:`, run `uv sync --extra cu128 --no-install-project`;
  3. then `uv run python infer.py --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small --no-ref ...`.

  This job is [`experiments/01_infer_fp32.yaml`](../experiments/01_infer_fp32.yaml).
- **Output:** a 48 kHz mono, **27.1 s** WAV.
  - Peak −2.4 dBFS, 82 % voiced: real speech, not silence.
  - Content: a Japanese narration introducing Databricks / Data Intelligence Platform / Lakehouse / Unity Catalog /
    Mosaic AI.
- **Timing (A10, fp32, cold):**

  | Stage | Time |
  |---|---|
  | tokenize | 2.7 ms |
  | duration prediction | 150 ms |
  | **RF sampling (40 steps)** | **2.5 s** |
  | decode | 0.31 s |
  | SilentCipher watermark | 0.34 s |
  | total to decode | 3.3 s |

### Pitfalls
1. **The AIR image has no FFmpeg.** `torchaudio.save` fails with `libavutil.so.57 not found`. Fall back to `soundfile`.
2. **Writing a WAV directly to `/Volumes/...` fails** (`libsndfile: System error`). libsndfile seeks while writing
   the WAV header, and the UC Volume FUSE mount rejects it. Fix: write to `/tmp`, then `cp` to the Volume
   (sequential writes are fine).
3. AIR presets `HF_HUB_ENABLE_HF_TRANSFER=1`, but the environment has no `hf_transfer`. Set it to `0`.

---

## 2. Model Serving ✅ (solved with env_pack GPU Express Deployment)

**It serves.** A Japanese text sent to the endpoint came back as audio:
- 48 kHz, 7.44 s;
- peak −1.9 dBFS, 76 % voiced;
- 3.51 s to synthesize.

- **Method:** wrap `InferenceRuntime` in an `mlflow.pyfunc.PythonModel` (models-from-code + `code_paths=[irodori_tts]`).
  Input is `text` plus parameters; output is a base64 WAV.
- **Standard GPU serving** came first and ended in `DEPLOYMENT_FAILED` (root cause below). Switching to **env_pack GPU
  Express Deployment** worked.

### 🔴 Root cause (an important product constraint)
From the serving build logs:
- **What happened during the build:**
  1. With `--extra-index-url .../cu128`, **torch 2.10.0+cu128 installs correctly**.
  2. The Model Serving GPU base image then **force-installs `torch 2.7.1+cu118` + torchvision** from its own package
     repo, uninstalling torch 2.10/cu128. The build log shows
     `Uninstalling torch-2.10.0+cu128 ... Successfully installed ... torch-2.7.1+cu118`.
  3. At load, `import torchaudio` (still cu128, needs libcudart.so.12) fails with
     **`OSError: libcudart.so.12: cannot open shared object file`**, and the model server does not start.
- **In short:**
  - AIR (training / batch) runs the torch 2.10/cu128 stack as is.
  - Managed GPU Model Serving pins torch 2.7.1 / CUDA 11.8 and overwrites a cu128 stack.
  - This model needs torch>=2.10, so standard GPU serving cannot host it today.

### ✅ The fix: env_pack GPU Express Deployment (adopted)
**`env_pack="databricks_model_serving"` packs the serverless GPU notebook's environment and restores it at serving,
without rebuilding the container**, so torch is not overwritten. The approach came from experience serving other
models, internal search, and the Databricks docs.

- **Registering:**
  - Run a notebook task with `databricks jobs submit`, using **`compute.hardware_accelerator="GPU_1xA10"` +
    `environments[].spec.client="4"`**. This means serverless GPU with a client image, which is required: classic
    compute fails with `is_client_image=False`.
  - `mlflow==3.12.0` is required.
  - Call `log_model` without `registered_model_name`, then
    `mlflow.register_model(env_pack=EnvPackConfig(name="databricks_model_serving", install_dependencies=False))`.
- **Result:** a UC model, then an endpoint `irodori-tts-express` (`GPU_MEDIUM` / A10, scale-to-zero), READY.
  The notebook is now [`notebooks/register_serving.py`](../notebooks/register_serving.py).

### Dependency traps hit with env_pack (all solved)
1. **protobuf.**
   - `dacvae` hard-imports `audiotools` (descript-audiotools 0.7.2), which pins an old `protobuf<3.20`.
   - protobuf 3.19.6 lacks `google.protobuf.internal.builder`, which **breaks the serverless notebook kernel**:
     pyspark / Spark Connect needs protobuf>=3.20. It also cannot coexist with `databricks-sdk` (protobuf>=4.25.8).
   - Fix, a **two-phase install**:
     1. install the real deps + `protobuf>=5.29.5,<6`;
     2. install `dacvae / silentcipher / descript-audiotools / argbind` with `--no-deps`.
   - audiotools only touches protobuf through tensorboard logging, which neither import nor inference needs; importing
     it on protobuf 5.x was verified.
2. **torchvision.**
   - The GPU client image ships a torchvision built for its base torch 2.7.1. Against torch 2.10 it is ABI-broken:
     `partially initialized module 'torchvision'...` when transformers probes it.
   - `%pip uninstall` cannot remove the base layer, so **install the matching torchvision from the cu128 index**.
     This only happens on the GPU image, not on CPU.
3. **Upload timeout.**
   - env_pack uploads a large `model_environment.tar` (the torch/cu128 stack). Without `databricks-sdk>=0.102.0` it
     hits a `TimeoutError` after 5 minutes. Installing descript-audiotools with `--no-deps` made the newer sdk
     installable.
   - Bundling the HF cache was dropped because it made the upload too big. The endpoint downloads from the Hub at load;
     the Express serving environment has egress.
4. **Git dependencies.** With them, env_pack's dependency step fails with `"Cannot find child https:"`. Fix:
   `install_dependencies=False`.
5. **Debugging tips.**
   - A notebook's `print` / `%pip` output does not appear in `jobs get-run-output`. Return diagnostics with
     `dbutils.notebook.exit(json.dumps(...))`.
   - Check dependency resolution first with `--dry-run` on fast **CPU serverless**, then run the ~18 min GPU
     registration.
   - Write WAVs in memory with `io.BytesIO()` to avoid the FUSE seek.

→ **Product feedback:** bleeding-edge OSS models that run on AIR do not fit standard GPU Model Serving, because of its
torch/CUDA pinning, but **env_pack Express Deployment can host them**. env_pack still has the traps above, and the
error messages and docs could be clearer:
- old protobuf pins in OSS dependencies vs. the serverless kernel's pyspark requirement;
- the torchvision shipped with the GPU image;
- the upload timeout of the large environment tar.

### General API design (v3) and the playback app
The API was redesigned so that an AI agent can tune the model later: **every model parameter can be set through the
API**.
- **Input** is `{text, params_json}`.
  - `params_json` accepts **all 43 fields of `SamplingRequest`**, read with `dataclasses.fields`, so it follows model
    updates.
  - It also accepts `ref_wav_b64` (base64 reference audio = voice cloning) and `{"__list_params__": true}`
    (self-description for agents).
  - The parameters are one JSON string column so that schema enforcement does not drop unknown keys.
  - The response returns `used_seed`, `messages` (runtime notes such as length prediction), `applied_params` and
    `error`.
- **Playback app**, [`app/tts_app.py`](../app/tts_app.py):
  - standard library only, auth from a CLI profile;
  - text in → synthesis → playback in the browser;
  - it calls REST `invocations` directly and waits 600 s, avoiding the CLI's 60 s timeout during a scale-to-zero cold
    start.
- **Long text (where the 30 s limit comes from):**
  - One synthesis is capped at `max_text_len≈256 tokens` + `max_seconds≈30 s`; longer text is cut off.
  - So the app **splits into sentences → synthesizes each → concatenates**. Measured: a webinar script → 60.1 s
    concatenated.
- **Voice consistency:**
  - With `--no-ref` the voice changes from sentence to sentence.
  - **Fixed voice** mode: generate one reference from the first sentence, then condition every sentence on it with a
    fixed seed.
  - In an A/B test the spread of per-sentence median F0 **halved**: stdev 19.9 → 10.8 Hz, range 64.8 → 28.0 Hz.
  - Passing any reference audio clones that voice.
- **Seam noise:**
  - **The problem:** one sentence per call gives short pieces (~2–5 s), so there are many seams, and non-zero cut
    points click.
  - **Fix 1:** group sentences into **~24 s chunks** (default 170 characters). A webinar script went from 8 sentences
    to **3 chunks / 2 seams**.
  - **Fix 2:** a **12 ms fade** at each clip edge, with the last sample set to 0.
  - **Why ~24 s:** **30 s is a structural limit of the model** (`max_latent_steps≈750` ÷ ≈25.2 frames/s ≈ 30 s), so
    aim just below it.
- **Semantic splitting (LLM / Databricks FMAPI):** splits a presentation script at meaning boundaries into ~24 s
  chunks.
  - **Safe by design:**
    1. the sentence split is done locally (`_atomic_units`);
    2. the LLM (default `databricks-claude-haiku-4-5`) only returns **the grouping** (a sequence of indices);
    3. the indices are checked to be a complete ordered partition, which **guarantees the read text verbatim** (zero
       edits).
  - On failure it falls back to character-count splitting.
  - Verified: a 469-character presentation → 5 topic chunks (intro / problem / architecture / demo / wrap-up,
    10.9–19.9 s each, no clamping).
  - The UI chooses between semantic (LLM) and character-count.

---

## 3. LoRA fine-tuning ✅

- **Method:** [`experiments/finetune/finetune_poc.py`](../experiments/finetune/finetune_poc.py) checks the whole
  pipeline end to end:
  1. generate 8 short clips with the base model;
  2. encode them to latents with the same DACVAE codec, and build a manifest (`text` + `latent_path` + `num_frames`);
  3. **train LoRA for 30 steps** with `train.py --config configs/train_v4_small_lora.yaml`;
  4. load the saved PEFT adapter with `--lora-adapter` and synthesize again.
- **Result:**
  - Training took **30 steps in 38.8 s**: ~9 samples/s, 2.27 step/s, loss 1.05 → 1.02, dur_mae improved.
  - The adapter (`adapter_model.safetensors`, 105 MB) was saved, reloaded, and used to synthesize 4.4 s of audio.
- **LoRA settings:**
  - r=16, alpha=32;
  - target = diffusion attention (`^blocks.\d+.attention.(wq|wk|wv|wo|...)$`);
  - `duration_predictor` in modules_to_save.
- **Note:**
  - 8 clips / 30 steps is a **plumbing check** (deliberate overfitting).
  - Real speaker cloning needs a real-data manifest (`prepare_manifest.py` builds it from an HF dataset) and thousands
    of steps.
  - Multi-GPU training (`serverless_gpu @distributed` / `torchrun`) was not tested.

---

## 4. Quantization performance ✅ (A10G / H100)

The script is [`experiments/bench_quant.py`](../experiments/bench_quant.py).
- **Setup:** each variant ran 5 times with the same text, seed and 40 steps.
- **RTF** = audio seconds / synthesis wall seconds. Higher is faster: a multiple of real time.

### A10G (compute capability 8.6)

| variant | ckpt (disk) | GPU weights | peak inference | model time | **RTF** |
|---|---|---|---|---|---|
| fp32 | 2922 MiB | 3199 MiB | 5085 MiB | 2.05 s | 9.7× |
| **bf16** | 2922 | 1768 | 3653 | **1.30 s** | **15.4×** ⚡ |
| int8-weight-only | 872 | 1154 | 3040 | 1.54 s | 13.0× |
| int8-dynamic | 872 | 1154 | 3055 | 9.55 s | 2.1× 🐢 |
| int4-weight-only | 813 | 1090 | 2990 | 2.21 s | 9.1× |
| float8-weight-only | 873 | 1155 | 3041 | 1.77 s | 11.3× |
| float8-dynamic | — | — | — | ❌ needs CC ≥ 8.9 | — |

### H100 (compute capability 9.0)

| variant | **RTF** |
|---|---|
| fp32 | 19.8× |
| bf16 | 19.0× |
| int8-weight-only | 15.7× |
| int8-dynamic | 2.4× 🐢 |
| int4-weight-only | 14.7× |
| float8-weight-only | 13.8× |
| float8-dynamic | 6.1× |

### Conclusions (important)
- **For this model (0.8B, 40 diffusion steps), quantization does nothing for speed.**
  - bf16 is fastest on both A10 and H100; on H100, fp32/TF32 matches it.
  - Weight-only quantization only adds dequantization overhead: the model is small, and compute is not
    memory-bandwidth bound.
- **Dynamic quantization (activation quantization, W8A8) clearly hurts.** int8-dynamic is 7× slower. float8-dynamic
  runs only on Hopper and is slower than bf16.
- **The real benefit of quantization is VRAM.**
  - int4 weights take **1090 MiB**: about 1/3 of fp32, about 60 % of bf16.
  - The checkpoint is 813 MiB on disk.
  - Use it to fit smaller GPUs or pack more serving replicas.
- **Hardware support:**
  - float8 needs **compute capability 8.9+** (Ada / Hopper). On an A10 (8.6) only weight-only works; dynamic does
    not.
  - int4-weight-only works on CC 8.0+, including the A10.

### Recommendations
- **Latency first** → bf16. It is 15× real time on A10 and 19× on H100, with no conversion.
- **VRAM / cost first (replica density)** → int4-weight-only or int8-weight-only. Slightly slower, far less memory.
- dynamic / float8-dynamic are not recommended for this model.

---

## 5. General lessons (they apply to other bleeding-edge OSS models too)

1. **For a repo with a `uv.lock`, let `command:` run `uv sync` instead of AIR's dependency resolution.**
   - Keep the environment minimal (just `uv`) and set `UV_PROJECT_ENVIRONMENT=/tmp/...`.
   - torch 2.10/cu128 runs fine on A10 and H100 (driver 580 / CUDA 13).
2. **Seeking writes to a UC Volume fail.** Write to `/tmp` and `cp`, or write in memory (`io.BytesIO`).
3. **The AIR image has no FFmpeg.** Do audio I/O with soundfile, or in memory.
4. **Inject extra packages (mlflow, ...) with `uv run --with`.** `uv pip install` goes into the base venv.
5. Registering a UC model needs `mlflow[databricks]` + boto3.
