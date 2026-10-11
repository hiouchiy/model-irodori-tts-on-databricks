# Serving API — `irodori-tts-express`（日本語音声合成）

このリポジトリでデプロイする Model Serving エンドポイントの API 仕様の正本。立ち上げ手順は [GUIDE.md](GUIDE.md) §4。

モデル: [`Aratako/Irodori-TTS-v4.1-Small`](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small)（0.8B、Flow Matching / RF-DiT、
DACVAE 48kHz コーデック、MIT）。env_pack（GPU Express Deployment）で、torch 2.10 / cu128 の環境をそのまま固めて
サービングしている（[`air_job/serving/serve_irodori_tts_express.py`](../air_job/serving/serve_irodori_tts_express.py)）。

## 0. 呼び出し方

```
POST https://<WORKSPACE_HOST>/serving-endpoints/<ENDPOINT>/invocations
Authorization: Bearer <TOKEN>
Content-Type: application/json
```

- **入力の形式**：MLflow PyFunc の `dataframe_records` を使う。**1 行が 1 発話**で、複数行を送ると上から順番に処理される（並列にはならない）。
- **出力の形式**：`{"predictions": [...]}` で、行ごとに要素が 1 つ返る。
- **エラーの返り方**：入力不正や推論失敗のときも HTTP 200 のまま返り、その行の `error` フィールドに理由が入る。HTTP 4xx / 5xx になるのは基盤側の問題のときだけ（認証、起動中、スキーマ不一致、タイムアウト）。
- **コールドスタート**：スケールゼロからの最初のリクエストは、A10 で約 405 秒かかった（実測）。
  - クライアントのタイムアウトは 600〜900 秒にする。
  - 503 / 504 / 429 が返ったら、60 秒待って再試行する。
  - `databricks serving-endpoints query` は内部のタイムアウトが約 60 秒なので、初回は失敗しやすい。REST で直接呼ぶ。
- **Model Serving 全般の仕様**（認証、入力形式など）は Databricks の公式ドキュメント
  [Query serving endpoints for custom models](https://docs.databricks.com/aws/en/machine-learning/model-serving/score-custom-model-endpoints) を参照。

## 1. リクエスト

| カラム | 型 | 必須 | 説明 |
|---|---|---|---|
| `text` | string | ✅ | 読み上げる日本語テキスト。**1 行で 1 発話**（上限は約 256 トークン / 約 30 秒）。長文は文単位に分けて複数行で送り、クライアント側でつなげる |
| `params_json` | string（JSON） | ✅（カラムとして） | 合成パラメータを JSON 文字列で渡す（下表）。`""` または `"{}"` なら既定値で合成する |

`params_json` を「1 つの JSON 文字列」にしているのは、MLflow のスキーマ強制で未知のキーが落とされないようにするため。
受け付けるキーはモデルの `SamplingRequest` から動的に取り出しているので、モデル側が更新されれば自動で追従する。

### パラメータ一覧の自己取得（合成しない）

```json
{"dataframe_records": [{"text": "", "params_json": "{\"__list_params__\": true}"}]}
```

→ レスポンスの `messages` に `{"supported_params": [...], "extras": ["ref_wav_b64"], "defaults": {...}}` が入る。
**既定値の正本はこの応答**（以下の表の既定値は v4.1-Small の実エンドポイントから取得した時点のもの）。

### `params_json` のキー

**話者・スタイル指定**

| キー | 型 | 既定 | 説明 |
|---|---|---|---|
| `ref_wav_b64` | string | — | **声クローン用の参照音声**（WAV を base64 化したもの）。この API 独自の拡張キー |
| `caption` | string | null | 声質・スタイルを指示するキャプション（例: `"落ち着いた、近い距離感の女性話者"`）。v4-Small は参照音声なし（キャプションのみ）でも、参照音声＋キャプションでも使える |
| `no_ref` | bool | （自動） | 参照音声なしで合成する。**参照系のキーが無いとき、この API が自動で `true` にする** |
| `ref_normalize_db` | float / null | -16.0 | 参照音声を、エンコード前にこのラウドネスへ正規化する。`null` で無効 |
| `ref_ensure_max` | bool | true | 参照音声のピークが 1.0 を超えたときだけ縮小する（正規化が無効のときに効く） |
| `max_ref_seconds` | float / null | null | 参照音声の最大長（秒）。null ならチェックポイントの推奨値。0 以下で上限なし |
| `ref_wav` / `ref_wavs` / `ref_latent` / `ref_latents` / `ref_embed` | path | null | **サーバ側のファイルパス**を指定するキー。クライアントからは使えないので `ref_wav_b64` を使う |
| `lora_adapter` | path | null | サーバ側の LoRA ディレクトリ。現行エンドポイントでは未使用 |

**長さ**

| キー | 型 | 既定 | 説明 |
|---|---|---|---|
| `seconds` | float / null | null | 出力の長さを秒で指定する。省略すると長さ予測器が自動で推定する（推奨） |
| `duration_scale` | float | 1.0 | 予測した長さに掛ける倍率（>1 で長く＝ゆっくり、<1 で短く＝速く） |
| `min_seconds` / `max_seconds` | float | 0.5 / 30.0 | 予測した長さのクランプ範囲 |
| `trim_tail` | bool | true | 末尾のほぼ無音の区間を切り詰める |
| `tail_window_size` / `tail_std_threshold` / `tail_mean_threshold` | int / float / float | 20 / 0.05 / 0.1 | 末尾トリムの判定パラメータ |

**サンプリング・品質**

| キー | 型 | 既定 | 説明 |
|---|---|---|---|
| `seed` | int / null | null（ランダム） | 乱数シード。指定すると再現できる。使ったシードはレスポンスの `used_seed` に返る |
| `num_steps` | int / null | null（RF は 40） | サンプリングのステップ数。減らすと速く、増やすと品質が安定する |
| `t_schedule_mode` | `"linear"` / `"sway"` | linear | タイムステップのスケジュール。`sway` は少ないステップ数（例 6）で高速化するための実験的な設定 |
| `sway_coeff` | float | -1.0 | `sway` 時の係数 |
| `cfg_scale_text` | float | 3.0 | テキスト条件の CFG 強度 |
| `cfg_scale_caption` | float | 3.0 | キャプション条件の CFG 強度 |
| `cfg_scale_speaker` | float | 5.0 | 話者（参照音声）条件の CFG 強度 |
| `cfg_scale` | float / null | null | 非推奨。指定すると上の 3 つをまとめて上書きする |
| `cfg_guidance_mode` | `"independent"` / `"joint"` / `"alternating"` | independent | CFG の方式 |
| `cfg_min_t` / `cfg_max_t` | float | 0.5 / 1.0 | CFG をかけるタイムステップの範囲 |
| `truncation_factor` | float / null | null | 初期ノイズのスケール（例 0.8 で平坦、0.9 でシャープ） |
| `rescale_k` / `rescale_sigma` | float / null | null | Temporal score rescaling（2 つは一緒に指定する） |
| `speaker_kv_scale` | float / null | null | >1 で話者らしさを強める（force-speaker） |
| `speaker_kv_min_t` / `speaker_kv_max_layers` | float / int | null | 上の話者強調をかけるタイムステップ・層の範囲 |
| `speaker_uncond_mode` | `"mask"` / `"noise"` | mask | 話者条件の無条件側の作り方 |
| `num_candidates` | int | 1 | 一度に生成する候補の数。**レスポンスには 1 本だけ返る**ので、通常は 1 のまま |
| `decode_mode` | `"sequential"` / `"batch"` | sequential | 複数候補をデコードする方式 |
| `context_kv_cache` | bool | true | テキスト・話者コンテキストの K/V を事前計算して高速化する |
| `max_text_len` / `max_caption_len` | int / null | null（256） | テキスト・キャプションの最大トークン長 |

型変換: `num_steps`, `seed`, `num_candidates`, `max_text_len`, `max_caption_len`, `speaker_kv_max_layers`,
`tail_window_size` はサーバ側で `int()` にキャストされる。未知のキーは無視される。

## 2. レスポンス（1 行ごと）

| フィールド | 型 | 説明 |
|---|---|---|
| `audio_wav_b64` | string | 合成音声（**WAV、48kHz、mono、16bit** を base64 化したもの）。エラー時は空 |
| `sample_rate` | int | 48000 |
| `audio_seconds` | float | 音声の長さ（秒） |
| `synth_seconds` | float | 合成にかかった時間（秒、GPU 上） |
| `used_seed` | int | 実際に使ったシード |
| `messages` | string（JSON 配列） | ランタイムからの通知（長さのクランプ等）。`__list_params__` のときはパラメータ一覧 |
| `applied_params` | string（JSON） | 実際に適用したパラメータ（参照音声は `"<ref_wav>"` に置き換えて表示） |
| `error` | string | 正常なら `""`。失敗時は `"<例外型>: <メッセージ>"` |

## 3. 例

```bash
curl -sS -X POST "https://<WORKSPACE_HOST>/serving-endpoints/irodori-tts-express/invocations" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"dataframe_records":[{"text":"こんにちは。データブリックスのモデルサービングから話しています。",
       "params_json":"{\"seed\":42,\"caption\":\"落ち着いた、近い距離感の女性話者\"}"}]}' \
  | python3 -c "import sys,json,base64;r=json.load(sys.stdin)['predictions'][0];print(r['error'] or r['audio_seconds']);open('out.wav','wb').write(base64.b64decode(r['audio_wav_b64']))"
```

声クローン（Python。`invoke()` は上の §0 の呼び出しを包んだ関数）:

```python
import base64, json
ref = base64.b64encode(open("speaker.wav", "rb").read()).decode()
res = invoke("irodori-tts-express", {"dataframe_records": [
    {"text": "今日はいい天気ですね。", "params_json": json.dumps({"ref_wav_b64": ref, "seed": 1})},
    {"text": "散歩に行きましょう。",   "params_json": json.dumps({"ref_wav_b64": ref, "seed": 1})},
]})
for i, r in enumerate(res["predictions"]):
    open(f"line{i}.wav", "wb").write(base64.b64decode(r["audio_wav_b64"]))
```

## 4. 性能・制約

- **実測**（A10、起動済み）：4.0 秒の音声を 3.6 秒で合成した。初回の RF サンプリング（cold）は約 2.5 秒。GPU メモリのピークは約 4〜5 GiB。
- **1 行あたりの上限**：約 30 秒・約 256 トークン。長文は文単位に分割して送る。
- **重みの取得**：起動時に HF Hub から取得する（サービング環境から外部への通信がある）。

## 5. 再登録

env_pack なので、登録はローカルからはできない。[`air_job/serving/serve_irodori_tts_express.py`](../air_job/serving/serve_irodori_tts_express.py) をワークスペースに取り込み、サーバレス GPU（`GPU_1xA10`、client `4`）のジョブとして実行する。登録先は `uc_model` パラメータで指定する。手順の詳細は [GUIDE.md](GUIDE.md) §4。
