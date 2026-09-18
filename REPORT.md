# Irodori-TTS on Databricks AI Runtime — PoC レポート

**対象モデル:** [Aratako/Irodori-TTS-v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small)（0.8B, Flow Matching / RF-DiT + ModernBERT テキストエンコーダ + DACVAE 48kHz コーデック、日本語専用）
**リポジトリ:** https://github.com/Aratako/Irodori-TTS
**実行環境:** Databricks AI Runtime (AIR CLI beta) / ワークスペース `e2-demo-field-eng` / カタログ `hiroshi.air_samples` / 出力 Volume `hiroshi.air_samples.tts`
**日付:** 2026-09-16

---

## 0. 結論サマリ

| フェーズ | 結果 | 一言 |
|---|---|---|
| 1. AIR上で推論・音声生成 | ✅ 成功 | A10で27秒のDatabricks紹介音声を生成。RFサンプリング2.5s（cold）|
| 2. Model Serving | ✅ **成功**（env_pack GPU Express Deployment）。標準GPUサービングはtorch競合で不可 → env_packで解決し、実際にエンドポイントで音声取得 | 下記に全経緯 |
| 3. ファインチューニング (LoRA) | ✅ 成功 | 30ステップを38.8秒、アダプタ保存→再推論まで通った |
| 4. 量子化パフォーマンス | ✅ 完了（A10 + H100） | **このモデルでは量子化は速度に効かない。効くのはVRAM削減** |

**最大の技術的ポイント:** 本モデルは `torch>=2.10 / cu128`・`transformers 5.x`・git依存（dacvae, silentcipher）という最先端スタックを要求し、AIRの `environment.dependencies`（env v4 は torch 2.7.1）では再現不能。**AIRジョブの `command:` 内で `uv sync` を実行し、リポジトリ同梱の `uv.lock` から著者の環境をそのまま再現する**アプローチが有効だった。AIR側の依存解決を回避できるため、他の最先端OSSモデルにも横展開できる汎用パターン。

---

## 1. AIR上での推論・音声生成 ✅

- **手法:** リポジトリを code_source としてスナップショット、`command:` で `uv sync --extra cu128 --no-install-project` → `uv run python infer.py --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small --no-ref ...`
- **成果物:** `outputs/databricks_intro_fp32.wav`（`/Volumes/hiroshi/air_samples/tts/databricks_intro_fp32.wav`）
  - 48kHz mono, **27.1秒**, peak −2.4dBFS, voiced率82% → 無音ではなく実音声であることを確認
  - 内容: Databricks / Data Intelligence Platform / Lakehouse / Unity Catalog / Mosaic AI を紹介する日本語ナレーション
- **タイミング（A10, fp32, cold）:** tokenize 2.7ms / duration予測 150ms / **RFサンプリング 2.5s（40ステップ）** / decode 0.31s / SilentCipher電子透かし 0.34s → total_to_decode 3.3s

### ハマりどころ
1. **AIRイメージに FFmpeg が無い** → `torchaudio.save` が `libavutil.so.57 not found` で失敗。`soundfile` にフォールバックする。
2. **`/Volumes/...` へ直接WAV書き込みが失敗**（`libsndfile: System error`）。libsndfile がWAVヘッダ書き込み時にランダムseekするが、UC Volume の FUSE マウントが拒否するため。**`/tmp` にローカル書き込み → `cp` で Volume に転送**（順次書き込みはFUSE OK）で解決。
3. AIRが `HF_HUB_ENABLE_HF_TRANSFER=1` をプリセットするが uv venv に `hf_transfer` が無い → `HF_HUB_ENABLE_HF_TRANSFER=0` を設定。

---

## 2. Model Serving ✅（env_pack GPU Express Deployment で解決）

**結論：ホストできました。** 実際にエンドポイント `irodori-tts-express` へ日本語テキストを投げ、音声（48kHz, 7.44秒, peak −1.9dBFS, voiced率76%, 合成3.51秒）を取得（[outputs/serving/endpoint_reply.wav](Irodori-TTS/../outputs/serving/endpoint_reply.wav)）。

- **手法:** `InferenceRuntime` を `mlflow.pyfunc.PythonModel` でラップ（models-from-code + `code_paths=[irodori_tts]`）。入力 `text`（+ caption/num_steps/seed）→ 出力 base64 WAV。
- 最初は**標準GPUサービング**を試したが `DEPLOYMENT_FAILED`（下記の根本原因）。→ **env_pack GPU Express Deployment** に切り替えて成功。

### 🔴 根本原因（重要な製品制約）
サービングのログから判明：
- ビルド時、私の `--extra-index-url .../cu128` で **torch 2.10.0+cu128 が正しく入る**が、その後に **Model Serving のGPUベースイメージが自前の package-repo から `torch 2.7.1+cu118` + torchvision を強制インストールし、私の torch 2.10/cu128 をアンインストール**してしまう（ビルドログに `Uninstalling torch-2.10.0+cu128 ... Successfully installed ... torch-2.7.1+cu118` と明記）。
- 結果、ロード時に `import torchaudio`（cu128のまま、libcudart.so.12を要求）が **`OSError: libcudart.so.12: cannot open shared object file`** で失敗 → モデルサーバ起動失敗。
- **要するに：AIR（学習/バッチ）は uv 経由で torch 2.10/cu128 の最先端スタックをそのまま動かせるが、マネージドGPU Model Serving は torch 2.7.1 / CUDA 11.8 をピン留めしており、cu128スタックを上書きしてしまう。** 本モデルは torch>=2.10 を要求するため、標準GPUサービングでは現状ホスト不可。

### ✅ 打開策：env_pack GPU Express Deployment（採用・成功）
Qwenサービングのノウハウ（`~/Documents/Work/20260330_ASQ_MIZUHO_Qwen35_Serving`）+ 社内Glean + Databricks docs から特定。**`env_pack="databricks_model_serving"` はサービング時のコンテナ再ビルドをせず、Serverless GPU ノートブックの環境をそのままパックして復元する**ため torch 上書きが起きない。
- **登録:** `databricks jobs submit` の notebook タスクを **`compute.hardware_accelerator="GPU_1xA10"` + `environments[].spec.client="4"`**（Serverless GPU / client image 必須。Classic は `is_client_image=False` で失敗）。`mlflow==3.12.0` 必須。`log_model`（`registered_model_name` を付けない）→ `mlflow.register_model(env_pack=EnvPackConfig(name="databricks_model_serving", install_dependencies=False))`。
- **成果:** UC `hiroshi.air_samples.irodori_tts_express`（v2）→ エンドポイント `irodori-tts-express`（`GPU_MEDIUM`/A10, scale-to-zero）READY。ノートブック: [air_job/serving/serve_irodori_tts_express.py](Irodori-TTS/air_job/serving/serve_irodori_tts_express.py)。

### env_pack でぶつかった依存地雷（全て解決。ログはノートブックの `env_report()`/`stage()` に出力）
1. **`dacvae` が `import audiotools`（descript-audiotools 0.7.2）を強制** → これが古い `protobuf<3.20` をピン。protobuf 3.19.6 は `google.protobuf.internal.builder` を欠き、**Serverlessノートブックのカーネル（pyspark/Spark-Connect, protobuf>=3.20必須）を破壊**。かつ `databricks-sdk`（protobuf>=4.25.8）とも両立不可。→ **2フェーズ install**: ①実依存 + `protobuf>=5.29.5,<6`、②`dacvae/silentcipher/descript-audiotools/argbind` を `--no-deps`（audiotools は tensorboard 経由でしか protobuf を触らず import/推論には不要 → protobuf 5.x でimport成功を検証）。
2. **GPU client image が base torch 2.7.1 用 torchvision を同梱** → torch 2.10 に対しABI不整合で `partially initialized module 'torchvision'...`（transformers probe時）。`%pip uninstall` は base層を消せない → **cu128 index から一致する torchvision を install**（CPUイメージでは再現せずGPUのみ発生）。
3. **env_pack は巨大な `model_environment.tar`（torch/cu128スタック）をアップロード** → `databricks-sdk>=0.102.0` が無いと5分で `TimeoutError`。descript-audiotools を `--no-deps` にしたことで sdk を入れられるように。HFキャッシュのバンドルはアップロード肥大化のため止め、サービング時にHubからDL（Express サービング環境は egress あり）。
4. git依存があると env_pack の dependency ステップが `"Cannot find child https:"` で失敗 → `install_dependencies=False` で回避。
5. **デバッグTips:** ノートブックの `print`/`%pip` 出力は `jobs get-run-output` に出ない → 診断は `dbutils.notebook.exit(json.dumps(...))` で返す。依存解決は**高速なCPU Serverless の `--dry-run`** で先に検証してから ~18分のGPU登録を回す。WAV書き込みは `io.BytesIO()` でインメモリ（FUSE seek回避）。

→ **EPLフィードバック:** 「AIRで動く最先端OSSモデルは、標準GPU Model Serving のtorch/CUDAピン留めではそのまま載らないが、**env_pack Express Deployment で載せられる**」。ただし env_pack でも上記の依存地雷（特に古いprotobufピンを持つOSS依存 × Serverlessカーネルのpyspark要件、GPUイメージのtorchvision同梱、巨大env tarのアップロードtimeout）があり、エラーメッセージ/ドキュメントの改善余地あり。

### 汎用API設計（v3）＋ 再生アプリ
将来 AIエージェントがチューニングする前提で、**モデルの全パラメータをAPIで指定可能**に再設計（UC v3、エンドポイントもv3にスワップ済み）。
- **入力** `{text, params_json}`。`params_json` は `SamplingRequest` の**全43フィールドを `dataclasses.fields` で動的に受理**（モデル更新に自動追従）。`ref_wav_b64`（base64参照音声＝ボイスクローン）と、`{"__list_params__":true}` によるパラメータ自己記述（エージェント向け）も対応。スキーマ強制で未知パラメータが落ちないよう、あえて1本のJSON文字列列に集約。レスポンスは `used_seed`/`messages`（長さ予測等のランタイム注記）/`applied_params`/`error` を返す。
- **再生アプリ** [app/tts_app.py](Irodori-TTS/../app/tts_app.py)（標準ライブラリのみ・CLIプロファイル認証）。テキスト入力→合成→ブラウザ再生。認証はCLIの60秒タイムアウトを避けるため **REST `invocations` を直接叩き600秒待ち**（scale-to-zeroのコールドスタート対応）。
- **長文（30秒上限の正体）:** 1回の合成は `max_text_len≒256トークン`＋`max_seconds≒30秒`が上限で、長文は途中で切れる。→ アプリが**文分割→各文合成→連結**（実測: ウェビナー原稿→60.1秒連結）。
- **声の一貫性:** `--no-ref`は文ごとに声が変わるため、**「声を固定」モード**（第1文から参照音声を1つ生成→全文をその参照＋固定seedで条件付け）を実装。A/B検証で文間の中央値F0のばらつきが **stdev 19.9→10.8Hz / range 64.8→28.0Hz** に半減（`outputs/serving/longform_drift.wav` vs `longform_fixed.wav`）。任意の参照音声を渡せば狙った声にクローンも可能。
- **継ぎ目ノイズ対策:** 1文ずつだと細切れ（〜2-5秒）で継ぎ目が多発＋切断点が非ゼロ→クリック音。→ (1) 文を **〜24秒（既定170文字）** の塊にグルーピング（ウェビナー原稿: 8文→**3チャンク/継ぎ目2**）、(2) 各クリップ端に **12msフェード**（最終サンプルを0に）。**30秒はモデルの構造的上限**（`max_latent_steps≈750`÷約25.2frame/s≈30s）なので、その手前を狙うのが最適。`outputs/serving/longform_smooth.wav` が現行ベスト。
- **セマンティック分割（LLM / Databricks FM API）:** プレゼン原稿を「意味の切れ目」で〜24秒チャンクに自動分割。**安全設計**として、文の分割は自前（`_atomic_units`）で行い、LLM（既定 `databricks-claude-haiku-4-5`）には**グループ化（番号の並び）だけ**を返させ、インデックスが完全な順序分割か検証→**読み上げテキストは逐語で保証**（改変ゼロ）。失敗時は文字数ベースへ自動フォールバック。検証: 469字プレゼン→話題単位の5チャンク（導入/課題/アーキ/デモ/まとめ、各10.9〜19.9秒・クランプ無し）[outputs/serving/presentation_semantic.wav](Irodori-TTS/../outputs/serving/presentation_semantic.wav)。UIで「セマンティック(LLM)／文字数ベース」を選択可。

---

## 3. ファインチューニング (LoRA) ✅

- **手法:** [air_job/finetune/finetune_poc.py](Irodori-TTS/air_job/finetune/finetune_poc.py) が完全自己完結でパイプライン疎通を検証。
  1. ベースモデルで8本の短いクリップを生成
  2. 同じDACVAEコーデックで latent にエンコード → manifest（`text` + `latent_path` + `num_frames`）を作成
  3. `train.py --config configs/train_v4_small_lora.yaml` で **30ステップのLoRA学習**
  4. 保存された PEFT アダプタを `--lora-adapter` で読み込み再合成
- **結果:** 学習 **30ステップ 38.8秒**（~9 samples/s, 2.27 step/s, loss 1.05→1.02, dur_mae改善）。アダプタ（`adapter_model.safetensors` 105MB）保存 → 再ロードして4.4秒音声の合成に成功。
- **LoRA設定:** r=16, alpha=32, target=diffusion attention（`^blocks.\d+.attention.(wq|wk|wv|wo|...)$`）, `duration_predictor` は modules_to_save。
- **注記:** 8クリップ/30ステップは**疎通確認用**（意図的な過学習）。実運用の話者クローン等は実データの manifest（`prepare_manifest.py` でHFデータセットから生成）と数千〜ステップが必要。マルチGPU学習（`serverless_gpu @distributed` / `torchrun`）は本PoCでは未検証。

---

## 4. 量子化パフォーマンス測定 ✅（A10G / H100）

同一テキスト・同一シード・40ステップで各バリアントを5回計測（[air_job/bench_quant.py](Irodori-TTS/air_job/bench_quant.py)）。RTF = 音声秒数 / 合成wall秒（大きいほど高速、リアルタイム比）。

### A10G（compute capability 8.6）

| variant | ckpt(disk) | GPU重み | peak推論 | model時間 | **RTF** |
|---|---|---|---|---|---|
| fp32 | 2922 MiB | 3199 MiB | 5085 MiB | 2.05s | 9.7× |
| **bf16** | 2922 | 1768 | 3653 | **1.30s** | **15.4×** ⚡ |
| int8-weight-only | 872 | 1154 | 3040 | 1.54s | 13.0× |
| int8-dynamic | 872 | 1154 | 3055 | 9.55s | 2.1× 🐢 |
| int4-weight-only | 813 | 1090 | 2990 | 2.21s | 9.1× |
| float8-weight-only | 873 | 1155 | 3041 | 1.77s | 11.3× |
| float8-dynamic | — | — | — | ❌ CC≥8.9必須 | — |

### H100（compute capability 9.0）

| variant | **RTF** |
|---|---|
| fp32 | 19.8× |
| bf16 | 19.0× |
| int8-weight-only | 15.7× |
| int8-dynamic | 2.4× 🐢 |
| int4-weight-only | 14.7× |
| float8-weight-only | 13.8× |
| float8-dynamic | 6.1× |

### 結論（重要）
- **このモデル（0.8B, 拡散40ステップ）では、量子化は"速度"にまったく効かない。** A10でもH100でも **bf16（またはH100ではfp32/TF32）が最速**。weight-only量子化は逐次dequantのオーバーヘッドが乗るだけ（モデルが小さく、演算がメモリ帯域律速ではない）。
- **dynamic量子化（activation量子化, W8A8）は明確に有害**（int8-dynamic は 7倍遅い）。float8-dynamic は Hopper でのみ動くが bf16 より遅い。
- **量子化の実利は VRAM 削減**：int4 で重み **1090 MiB（fp32の約1/3、bf16の約6割）**、チェックポイントもディスク 813 MiB。→ より小さいGPUに載せる／サービングのレプリカ密度を上げる用途に有効。
- float8 は **compute capability 8.9+（Ada/Hopper）必須**。A10（8.6）では weight-only のみ動作、dynamic は不可。int4-weight-only は CC 8.0+ で A10 でも動作可。

### 推奨
- **レイテンシ最優先** → bf16（A10で15×、H100で19×リアルタイム。追加変換不要）。
- **VRAM/コスト最優先（レプリカ密度）** → int4-weight-only または int8-weight-only。速度は多少落ちるがメモリを大幅削減。
- dynamic / float8-dynamic は本モデルでは非推奨。

---

## 5. 汎用的な教訓（他の最先端OSSモデルにも適用可）

1. **`uv.lock` を持つ最先端リポジトリは、AIRの依存解決に載せず `command:` で `uv sync` させる。** env は最小（`uv` のみ）、`UV_PROJECT_ENVIRONMENT=/tmp/...` を指定。torch 2.10/cu128 も A10/H100 で問題なく動作（driver 580 / CUDA 13）。
2. **UC Volume への seek 依存の書き込みは失敗する。** `/tmp` に書いて `cp`、またはインメモリ（`io.BytesIO`）。
3. **AIRイメージに FFmpeg は無い。** 音声IOは soundfile / インメモリで回避。
4. **`uv run --with` で追加パッケージ（mlflow等）を注入**。`uv pip install` は base venv に入るので注意。
5. UC モデル登録は `mlflow[databricks]` + boto3 が必要。

## 6. 生成物一覧（ローカル `outputs/`）
- `databricks_intro_fp32.wav` — Phase1 の27秒Databricks紹介音声
- `bench_a10/*.wav`, `bench_h100/*.wav` — 各量子化バリアントの生成音声（聴き比べ用）+ `results.json`
- `finetune/after_lora.wav` — LoRA適用後の合成音声 / `manifest.jsonl` / `adapter_config.json`
- `serving/pyfunc_reload_sample.wav` — pyfunc再ロード検証の合成音声
