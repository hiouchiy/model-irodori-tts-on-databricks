# Serving API — Irodori-TTS (Japanese text-to-speech)

The single source for the API of the Model Serving endpoints this repo deploys (e.g. `irodori-tts-express` for v4.1-Small,
`irodori-tts-large` for v4-Large). How to register and deploy them: [README](../README.md) "3. Model Serving"; which
model to choose: README "Choosing a model".

Models: [`Aratako/Irodori-TTS-v4.1-Small`](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) (0.8B, MIT) or
[`Aratako/Irodori-TTS-v4-Large`](https://huggingface.co/Aratako/Irodori-TTS-v4-Large) (3.29B, Gemma Terms of Use), both
flow matching RF-DiT with a DACVAE 48 kHz codec. **The API is the same for both**; one endpoint serves one model
(fixed at registration). The accepted `params_json` keys and their defaults come from the loaded model, so ask each
endpoint with `__list_params__`. It is served with env_pack (GPU Express Deployment), which restores the torch 2.10 /
cu128 environment it was registered from. The PyFunc is [`serving/irodori_pyfunc.py`](../serving/irodori_pyfunc.py);
the request handling is `irodori_core.Synthesizer` ([`irodori_core.py`](../irodori_core.py)).

## 0. Calling it

```
POST https://<WORKSPACE_HOST>/serving-endpoints/<ENDPOINT>/invocations
Authorization: Bearer <TOKEN>
Content-Type: application/json
```

- **Input format:** MLflow PyFunc `dataframe_records`. **One row is one utterance.** Rows are processed in order, not in
  parallel.
- **Output format:** `{"predictions": [...]}`, one element per row.
- **Errors:**
  - A synthesis failure still returns HTTP 200, with the reason in that row's `error`.
  - HTTP 4xx / 5xx come from the platform: auth, starting up, schema mismatch, timeout.
  - A `params_json` that is not valid JSON, or a non-integer for an integer key, fails the whole request.
- **Cold start:** the first request after scale-to-zero took about **405 s** on an A10 (measured).
  - Use a client timeout of 600–900 s.
  - On 503 / 504 / 429, wait 60 s and retry.
  - `databricks serving-endpoints query` times out after about 60 s, so the first call often fails; call REST directly.
- **Model Serving in general** (auth, input formats): Databricks docs,
  [Query serving endpoints for custom models](https://docs.databricks.com/aws/en/machine-learning/model-serving/score-custom-model-endpoints).

## 1. Request

| Column | Type | Required | Description |
|---|---|---|---|
| `text` | string | ✅ | Japanese text to read. **One utterance per row** (limit ≈ 256 tokens / ≈ 30 s). Split long text into sentences, send several rows, and concatenate on the client. |
| `params_json` | string (JSON) | ✅ (as a column) | Synthesis parameters as a JSON string (table below). `""` or `"{}"` uses the defaults. |

`params_json` is a single JSON string so that MLflow schema enforcement never drops unknown keys. The accepted keys
are read from the model's `SamplingRequest` at load, so they follow model updates automatically.

### Listing the parameters (no synthesis)

```json
{"dataframe_records": [{"text": "", "params_json": "{\"__list_params__\": true}"}]}
```

The response's `messages` holds `{"supported_params": [...], "extras": ["ref_wav_b64"], "defaults": {...}}`. **This
response is the source for the defaults**; the defaults in the tables below were read from a live v4.1-Small endpoint.

### `params_json` keys

**Speaker and style**

| Key | Type | Default | Description |
|---|---|---|---|
| `ref_wav_b64` | string | — | **Reference audio for voice cloning** (a WAV, base64). An extension of this API. |
| `caption` | string | null | Caption describing the voice and style (e.g. `"落ち着いた、近い距離感の女性話者"`). v4-Small works with a caption only, or with reference audio + caption. |
| `no_ref` | bool | (auto) | Synthesize without reference audio. **This API sets it to `true` when no reference key is given.** |
| `ref_normalize_db` | float / null | -16.0 | Normalize the reference to this loudness before encoding. `null` disables it. |
| `ref_ensure_max` | bool | true | Scale the reference down only if its peak exceeds 1.0 (applies when normalization is off). |
| `max_ref_seconds` | float / null | null | Maximum reference length in seconds. null = the checkpoint's recommendation; ≤ 0 = no limit. |
| `ref_wav` / `ref_wavs` / `ref_latent` / `ref_latents` / `ref_embed` | path | null | **Server-side file paths.** Not usable from a client; use `ref_wav_b64`. |
| `lora_adapter` | path | null | Server-side LoRA directory. Not used by the current endpoint. |

**Length**

| Key | Type | Default | Description |
|---|---|---|---|
| `seconds` | float / null | null | Output length in seconds. Omit it to let the duration predictor decide (recommended). |
| `duration_scale` | float | 1.0 | Multiplier on the predicted length (> 1 longer = slower, < 1 shorter = faster). |
| `min_seconds` / `max_seconds` | float | 0.5 / 30.0 | Clamp range for the predicted length. |
| `trim_tail` | bool | true | Trim the near-silent tail. |
| `tail_window_size` / `tail_std_threshold` / `tail_mean_threshold` | int / float / float | 20 / 0.05 / 0.1 | Tail-trim detection parameters. |

**Sampling and quality**

| Key | Type | Default | Description |
|---|---|---|---|
| `seed` | int / null | null (random) | Random seed; set it to reproduce. The seed used comes back in `used_seed`. |
| `num_steps` | int / null | null (RF: 40) | Sampling steps. Fewer is faster; more is more stable. |
| `t_schedule_mode` | `"linear"` / `"sway"` | linear | Timestep schedule. `sway` is an experimental setting for few steps (e.g. 6). |
| `sway_coeff` | float | -1.0 | Coefficient for `sway`. |
| `cfg_scale_text` | float | 3.0 | CFG strength of the text condition. |
| `cfg_scale_caption` | float | 3.0 | CFG strength of the caption condition. |
| `cfg_scale_speaker` | float | 5.0 | CFG strength of the speaker (reference) condition. |
| `cfg_scale` | float / null | null | Deprecated. Overrides the three above at once. |
| `cfg_guidance_mode` | `"independent"` / `"joint"` / `"alternating"` | independent | CFG mode. |
| `cfg_min_t` / `cfg_max_t` | float | 0.5 / 1.0 | Timestep range where CFG applies. |
| `truncation_factor` | float / null | null | Initial noise scale (e.g. 0.8 flatter, 0.9 sharper). |
| `rescale_k` / `rescale_sigma` | float / null | null | Temporal score rescaling (set both together). |
| `speaker_kv_scale` | float / null | null | > 1 strengthens speaker likeness (force-speaker). |
| `speaker_kv_min_t` / `speaker_kv_max_layers` | float / int | null | Timestep / layer range of the speaker boost above. |
| `speaker_uncond_mode` | `"mask"` / `"noise"` | mask | How the unconditional side of the speaker condition is built. |
| `num_candidates` | int | 1 | Candidates generated at once. **Only one comes back**, so leave it at 1. |
| `decode_mode` | `"sequential"` / `"batch"` | sequential | How several candidates are decoded. |
| `context_kv_cache` | bool | true | Precompute the text / speaker context K/V for speed. |
| `max_text_len` / `max_caption_len` | int / null | null (256) | Maximum token length of the text / caption. |

Type conversion: `num_steps`, `seed`, `num_candidates`, `max_text_len`, `max_caption_len`, `speaker_kv_max_layers`
and `tail_window_size` are cast with `int()` on the server. Unknown keys are ignored.

## 2. Response (one per row)

| Field | Type | Description |
|---|---|---|
| `audio_wav_b64` | string | The audio: **WAV, 48 kHz, mono, 16-bit**, base64. Empty on error. |
| `sample_rate` | int | 48000 |
| `audio_seconds` | float | Audio length in seconds |
| `synth_seconds` | float | Synthesis time in seconds (on the GPU) |
| `used_seed` | int | The seed actually used |
| `messages` | string (JSON array) | Runtime notes (e.g. length clamping). For `__list_params__`, the parameter list. |
| `applied_params` | string (JSON) | The parameters applied (a reference shows as `"<ref_wav>"`) |
| `error` | string | `""` on success; `"<ExceptionType>: <message>"` on failure |

## 3. Examples

```bash
curl -sS -X POST "https://<WORKSPACE_HOST>/serving-endpoints/<ENDPOINT>/invocations" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"dataframe_records":[{"text":"こんにちは。データブリックスのモデルサービングから話しています。",
       "params_json":"{\"seed\":42,\"caption\":\"落ち着いた、近い距離感の女性話者\"}"}]}' \
  | python3 -c "import sys,json,base64;r=json.load(sys.stdin)['predictions'][0];print(r['error'] or r['audio_seconds']);open('out.wav','wb').write(base64.b64decode(r['audio_wav_b64']))"
```

Voice cloning (Python; `invoke()` wraps the call in §0):

```python
import base64, json
ref = base64.b64encode(open("speaker.wav", "rb").read()).decode()
res = invoke("<ENDPOINT>", {"dataframe_records": [
    {"text": "今日はいい天気ですね。", "params_json": json.dumps({"ref_wav_b64": ref, "seed": 1})},
    {"text": "散歩に行きましょう。",   "params_json": json.dumps({"ref_wav_b64": ref, "seed": 1})},
]})
for i, r in enumerate(res["predictions"]):
    open(f"line{i}.wav", "wb").write(base64.b64decode(r["audio_wav_b64"]))
```

## 4. Performance and limits

- **Measured** (A10, warm): 4.0 s of audio in 3.6 s. The first RF sampling (cold) takes about 2.5 s. GPU memory peak is
  about 4–5 GiB.
- **Per row:** about 30 s / 256 tokens. Split long text into sentences.
- **Weights:** downloaded from the HF Hub at load, so the serving environment needs outbound access.

## 5. Re-registering

env_pack packs the environment of a serverless GPU notebook, so registration cannot run locally. Run
[`notebooks/register_serving.py`](../notebooks/register_serving.py) as a serverless GPU job (`GPU_1xA10`, client
`4`); the `uc_model` parameter sets the target. Steps: [README](../README.md) "3. Model Serving".
