# Irodori-TTS on Databricks — Japanese text-to-speech on one A10 GPU (v4.1-Small or v4-Large)

## What this is

Run the Japanese TTS model **[Irodori-TTS](https://github.com/Aratako/Irodori-TTS)** on **Databricks**. It is a flow
matching RF-DiT with a DACVAE 48 kHz codec, in two sizes; pick one per endpoint ([Choosing a model](#choosing-a-model)):
- [`Aratako/Irodori-TTS-v4.1-Small`](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small), 0.8B, MIT;
- [`Aratako/Irodori-TTS-v4-Large`](https://huggingface.co/Aratako/Irodori-TTS-v4-Large), 3.29B, Gemma Terms of Use.

One API exposes the whole model:

- any field of upstream `SamplingRequest` (~43 knobs: seed, caption / voice design, steps, CFG, duration, ...) via
  `params_json`, discovered dynamically with `{"__list_params__": true}`;
- **voice cloning** from a reference WAV (`ref_wav_b64`).

One request row is one utterance (~256 tokens / ~30 s). Long text is split and concatenated by the caller; the apps in
[`app/`](app/) do that.

It needs torch 2.10 / CUDA 12.8, which standard GPU Model Serving overwrites with its own torch. The endpoint is
therefore registered with **env_pack (GPU Express Deployment)**, which restores the registering notebook's
environment as is ([docs/REPORT.md](docs/REPORT.md) §2).

**Licence:** the code and v4.1-Small are MIT (Aratako). v4-Large is under the **Gemma Terms of Use**, because its
text encoder is derived from Google's T5Gemma 2. See [Licence](#licence).

## Choosing a model

Both run on the same code, the same runtime and the same API. The model is chosen once per registration (the
`model` parameter of `notebooks/register_serving`), and each gets its own UC model and endpoint.

| | **v4.1-Small** (default) | **v4-Large** |
|---|---|---|
| Size | 0.8B | 3.29B: 24-layer, 2048-dim DiT, T5Gemma 2 text/caption encoder |
| Strengths (upstream model cards) | **Kanji reading** (Joyo reading accuracy 93.4 % vs 92.8 %, JSUT Kana-CER 3.43 % vs 3.67 %) | **Voice design**: caption adherence 4.30 vs 4.23 of 5. **Voice cloning**: speaker similarity at every reference length; references up to 120 s |
| Choose it for | Narration where reading accuracy matters (slides, documents) | Voices designed from a caption, cloning from ~30 s or more of reference audio, emoji style control |
| Licence | MIT | Gemma Terms of Use + Prohibited Use Policy, and the model card's own restrictions (no impersonation without consent, no deceptive deepfakes) |
| `model` value | `v4.1-small` | `v4-large` |
| Endpoint here | `irodori-tts-express` | `irodori-tts-large` |
| GPU | A10 | A10: peak 12.8 GiB in bf16 (measured 2026-10-14) |

Revisions are pinned in `irodori_core.MODELS`. Both need the same upstream commit (`89f9d8f` added v4-Large support).

## Ways to run

| | Way | Entry point | Use it for |
|---|---|---|---|
| 1 | **Notebook on serverless GPU** | [`notebooks/generate.py`](notebooks/generate.py) | one utterance at a time, interactively |
| 2 | **Batch with the AI Runtime CLI** | [`air_job/generate.yaml`](air_job/generate.yaml) (`air run`) | an unattended job that writes to a Volume |
| 3 | **Model Serving** (GPU endpoint) | [`notebooks/register_serving.py`](notebooks/register_serving.py), [`serving/`](serving/) | a REST endpoint for apps; API in [docs/SERVING-API.md](docs/SERVING-API.md) |

All three use the same core, [`irodori_core.py`](irodori_core.py). It holds:
- the runtime pins and why they are what they are;
- the pinned upstream commit and the checkpoints (`MODELS`);
- `prepare` (clone);
- `Synthesizer`: the request contract, the serving endpoint's request handling.

The notebooks and the AIR job install the runtime with the same [`scripts/install.sh`](scripts/install.sh).

## Prerequisites and configuration

| Requirement | Notes |
|---|---|
| A **Databricks workspace** | **Unity Catalog**, **serverless GPU / AI Runtime** (A10) and serverless jobs, and **GPU Model Serving** (`GPU_MEDIUM` = A10). |
| **Permissions** | `CREATE MODEL` on a UC schema, write to a Volume, create serving endpoints. |
| **Databricks CLI** (`databricks`) | authenticated: `databricks auth login --host https://<workspace> --profile <PROFILE>`. |
| **AI Runtime CLI** (`air`) | Only for way 2. |

Placeholders used below — replace them with your values:

| Placeholder | Meaning | Where it is set |
|---|---|---|
| `<PROFILE>` | CLI profile | every `--profile` |
| `<CATALOG>.<SCHEMA>` | where the model is registered and the Volume lives | the `uc_model` parameter of `notebooks/register_serving` |
| `<VOLUME>` | UC Volume for generated audio: `/Volumes/<CATALOG>/<SCHEMA>/<VOLUME>` | the notebook's `output_dir` widget; [`air_job/generate.yaml`](air_job/generate.yaml) `DEST` |
| `<ENDPOINT>` | serving endpoint name, e.g. `irodori-tts-express` | way 3 |
| `<you>` | your workspace user name (`databricks current-user me`) | workspace paths |

```bash
databricks volumes create <CATALOG> <SCHEMA> <VOLUME> MANAGED --profile <PROFILE>   # once
```

Layout:

```
irodori_core.py                 runtime pins, pinned upstream, checkpoints (MODELS); prepare / Synthesizer (shared by 1–3)
scripts/install.sh              two-phase runtime install on serverless GPU (notebooks and the AIR job)
notebooks/generate.py           way 1
notebooks/test_serving_model.py runs serving/irodori_pyfunc.py in-process on a GPU, before registering
notebooks/register_serving.py   way 3: install → smoke test → log + register to UC with env_pack
air_job/generate.yaml           way 2;  generate.py is the CLI it runs
serving/irodori_pyfunc.py       the MLflow PyFunc (models-from-code): the whole API
app/tts_app.py                  local browser app that calls the endpoint (long text, fixed voice)
app/slides_narrate.py           Google Slides speaker notes → narrated MP4, through the endpoint
examples/databricks_text.txt    sample input
experiments/                    PoC AIR jobs: upstream uv.lock inference, LoRA fine-tune, quantization bench
docs/SERVING-API.md             endpoint API (single source)
docs/REPORT.md                  PoC report: design decisions, pitfalls, benchmarks
```

## 1. Notebook on serverless GPU

1. Put the repo in the workspace, so the notebook has `../irodori_core.py` and `../scripts/install.sh` next to it:
   - **Git folder** (recommended): *Workspace → Create → Git folder* with this repo's URL; or
   - **sync** a local clone: `databricks sync . /Workspace/Users/<you>/model-irodori-tts-on-databricks --profile <PROFILE>`.
2. Open `notebooks/generate` and attach it to **serverless GPU** compute: accelerator **A10**, environment version **4**.
3. Fill in the widgets and *Run all*.
   - `model`: `v4.1-small` or `v4-large` ([Choosing a model](#choosing-a-model)).
   - `text` is one utterance.
   - `params_json` takes **any** `params_json` key of the endpoint, e.g. `seed`, `caption`, `num_steps`,
     `duration_scale`.
   - `output_dir` is a folder in your Volume.

   The first cell installs the runtime and restarts Python. The pinned upstream is cloned to `work_dir`
   (default `/tmp/irodori`). The WAV plus a `.json` with the response fields are written to `output_dir`, and the
   audio plays inline.

The same notebook runs unattended as a job on serverless GPU:

```bash
databricks jobs submit --no-wait --profile <PROFILE> --json '{
  "run_name": "irodori-generate",
  "tasks": [{"task_key": "generate",
             "notebook_task": {"notebook_path": "/Workspace/Users/<you>/model-irodori-tts-on-databricks/notebooks/generate",
                               "base_parameters": {"output_dir": "/Volumes/<CATALOG>/<SCHEMA>/<VOLUME>/irodori"}},
             "environment_key": "gpu", "compute": {"hardware_accelerator": "GPU_1xA10"}}],
  "environments": [{"environment_key": "gpu", "spec": {"client": "4"}}]}'
```

`environments[].spec.client: "4"` is required for serverless GPU.

## 2. Batch with the AI Runtime CLI

Set `DEST` in [`air_job/generate.yaml`](air_job/generate.yaml) and the `generate.py` flags in its `command`, then:

```bash
COPYFILE_DISABLE=1 air run --file air_job/generate.yaml --watch --profile <PROFILE>
```

The job:
1. runs `scripts/install.sh`;
2. runs [`generate.py`](generate.py), which clones the pinned upstream and synthesizes one utterance;
3. copies the WAV (and its `.json`) to the Volume.

`generate.py --text … | --text-file …` reads the input. `--model v4-large` picks the checkpoint (default
`v4.1-small`). `--params '{...}'` takes any `params_json` key of the endpoint.

## 3. Model Serving

```bash
# 1) put the repo in the workspace (as in way 1), then register (≈ 10–20 min on serverless GPU A10).
#    "model": "v4.1-small" (default) or "v4-large"; give each its own uc_model, e.g. ...irodori_tts_large.
databricks jobs submit --no-wait --profile <PROFILE> --json '{
  "run_name": "irodori-tts-register",
  "tasks": [{"task_key": "register",
             "notebook_task": {"notebook_path": "/Workspace/Users/<you>/model-irodori-tts-on-databricks/notebooks/register_serving",
                               "base_parameters": {"model": "v4.1-small", "uc_model": "<CATALOG>.<SCHEMA>.irodori_tts_express"}},
             "environment_key": "gpu", "compute": {"hardware_accelerator": "GPU_1xA10"}}],
  "environments": [{"environment_key": "gpu", "spec": {"client": "4"}}]}'
databricks jobs get-run <run_id> --profile <PROFILE>              # wait for TERMINATED/SUCCESS
databricks jobs get-run-output <task_run_id> --profile <PROFILE>  # notebook_output.result: {"uc_model", "model", "version", "smoke"}

# 2) endpoint (GPU_MEDIUM = A10, the same GPU family the environment was packed on)
databricks serving-endpoints create --no-wait --profile <PROFILE> --json '{
  "name": "<ENDPOINT>",
  "config": {"served_entities": [{"name": "irodori_tts_express-<VERSION>",
               "entity_name": "<CATALOG>.<SCHEMA>.irodori_tts_express", "entity_version": "<VERSION>",
               "workload_type": "GPU_MEDIUM", "workload_size": "Small", "scale_to_zero_enabled": true}],
             "traffic_config": {"routes": [{"served_model_name": "irodori_tts_express-<VERSION>", "traffic_percentage": 100}]}}}'
databricks serving-endpoints get <ENDPOINT> --profile <PROFILE> | jq '.state'   # READY / NOT_UPDATING

# 3) use it
python3 app/tts_app.py --profile <PROFILE> --endpoint <ENDPOINT>     # browser app at http://127.0.0.1:8765
```

To move an existing endpoint to a new version, use `databricks serving-endpoints update-config` with the same
`served_entities`. Run `notebooks/test_serving_model` first after changing `serving/` or `irodori_core.py`; it loads
the PyFunc in-process on a GPU without touching any endpoint.

**Calling the endpoint** ([docs/SERVING-API.md](docs/SERVING-API.md) is the full contract):

```bash
curl -sS -X POST "https://<workspace>/serving-endpoints/<ENDPOINT>/invocations" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"dataframe_records":[{"text":"こんにちは。","params_json":"{\"seed\":42}"}]}'
# -> {"predictions": [{"audio_wav_b64": ..., "sample_rate": 48000, "audio_seconds": ..., "error": ""}]}
```

The first request after scale-to-zero takes minutes; `databricks serving-endpoints query` gives up after ~60 s, so
call the REST API with a 600–900 s timeout (the apps do).

**Apps** (Python standard library only; auth from the CLI profile):
- [`app/tts_app.py`](app/tts_app.py) — text in, audio out in the browser. Splits long text into ~24 s chunks, by
  character count or semantically with an FMAPI LLM (default `databricks-claude-haiku-4-5`). *Fixed voice* makes one
  reference from the first chunk and conditions every chunk on it, so the voice does not drift. It applies 12 ms fades
  at the seams.
- [`app/slides_narrate.py`](app/slides_narrate.py) — reads each Google Slides speaker note and renders one narrated
  MP4. It needs ffmpeg and a Google OAuth token helper (`GOOGLE_AUTH_SCRIPT`):

  ```bash
  python3 app/slides_narrate.py --url "https://docs.google.com/presentation/d/<ID>/edit" --copy --clean --pages 60 --out deck.mp4
  ```

  - `--copy` works on a Drive copy.
  - `--clean` has an LLM rewrite hard kanji / English / product names into katakana before synthesis.

## Measured results

1× A10G, bf16, v4.1-Small unless noted ([docs/REPORT.md](docs/REPORT.md) has the details):

| | |
|---|---|
| Endpoint, warm | 4.0 s of audio in 3.6 s; 7.4 s of audio in 3.5 s. GPU memory peak ≈ 4–5 GiB |
| Cold start (scale-from-zero) | ≈ 405 s |
| Model time, 40 steps | bf16 1.30 s (RTF 15.4×), fp32 2.05 s (9.7×). Quantization saves VRAM, not time |
| Registration job (env_pack) | ≈ 10 min (2026-10-13; 15–20 min in the PoC) |
| **v4-Large**, endpoint `GPU_MEDIUM` (A10), warm | 7.0 s of audio in 5.3 s; 8.2 s in 4.4 s. Peak GPU memory 12.8 GiB (2026-10-14) |

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Endpoint fails with `libcudart.so.12: cannot open shared object file` | Standard GPU serving replaced torch 2.10/cu128 with 2.7.1/cu118. Register with env_pack (`notebooks/register_serving`), not plain `log_model`. |
| env_pack: `is_client_image=False` | Registration ran on classic compute. Use serverless GPU with `environments[].spec.client: "4"`. |
| env_pack dependency step: `Cannot find child https:` | Git dependencies. Register with `install_dependencies=False` (the notebook does). |
| `TimeoutError` uploading `model_environment.tar` | `databricks-sdk` older than 0.102.0. It is pinned in `irodori_core.RUNTIME_REQUIREMENTS`. |
| Notebook kernel breaks after install (`google.protobuf.internal.builder`) | descript-audiotools pulled protobuf<3.20. It is installed `--no-deps` in phase 2 of `scripts/install.sh`. |
| `partially initialized module 'torchvision'` | The GPU image's torchvision is built for torch 2.7.1. `torchvision==0.25.0` (for torch 2.10) is pinned. |
| `ValueError` about `hf_transfer` on download | Serverless GPU presets `HF_HUB_ENABLE_HF_TRANSFER=1` without the package. `irodori_core` sets it to `0`. |
| `libsndfile: System error` writing a WAV to `/Volumes/...` | The Volume mount rejects seeks. Write to `/tmp` (or in memory) and copy. |
| `serving-endpoints query` fails after ~60 s | Cold start. Retry later, or call REST with a long timeout. |
| Audio cut at ~30 s | One utterance is capped at ~256 tokens / ~30 s. Split long text (the apps do). |
| English words or rare kanji misread | Rewrite them in katakana (`slides_narrate.py --clean` does). |
| A notebook's `print` is missing from `jobs get-run-output` | Serverless only returns `dbutils.notebook.exit`; the notebooks return their results that way. |

Upstream is pinned to Irodori-TTS commit `89f9d8fbd4d51ea019867ee1197725ede1df13c5` (`irodori_core.UPSTREAM_SHA`).

## Licence

- **This repo's code:** MIT, like upstream (see [NOTICE.md](NOTICE.md)).
- **Irodori-TTS code and v4.1-Small weights:** MIT, © 2026 Aratako.
- **v4-Large weights:** the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Its shared text/caption encoder
  is derived from `google/t5gemma-2-1b-1b`. In short (not legal advice; read the terms):
  - **Commercial use is allowed.** Google claims no rights in the outputs (§3.3); you are responsible for them.
  - **Your own use** (including your organisation's internal endpoint) must follow the use restrictions of §3.2, i.e.
    the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
  - **Serving it to third parties is "Distribution":** the terms count a hosted service (API, web) for a third party
    as Distribution. Then put the §3.2 use restrictions into the terms of use your users agree to, and tell them the
    model is subject to them.
  - **Copies need notice:** pass on the terms. Copies other than through a hosted service need a `NOTICE` file:
    "Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms".
  - **Google may restrict** usage it believes violates the terms.
  - **The model card adds:** no cloning or impersonation of a real person's voice without their consent, no deceptive
    deepfakes or misinformation.
- Check the [repo](https://github.com/Aratako/Irodori-TTS) and the model cards
  ([Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small), [Large](https://huggingface.co/Aratako/Irodori-TTS-v4-Large))
  for current terms before use.
- Upstream embeds an inaudible **SilentCipher** watermark in generated audio.
