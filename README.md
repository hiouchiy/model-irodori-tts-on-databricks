# Irodori-TTS on Databricks

日本語音声合成 OSS **[Irodori-TTS](https://github.com/Aratako/Irodori-TTS)**（Aratako 氏, MIT）を
**Databricks** 上で動かし、**Model Serving の GPU エンドポイント**として提供するための一式です。

> **これ一つでできること**
> 1. **モデルのサービング** — Unity Catalog に登録し、GPU エンドポイントで REST 提供（本命）
> 2. **ローカル再生アプリ** — テキスト → 音声をブラウザで生成・再生（`app/tts_app.py`）
> 3. **スライド → ナレーション動画** — Google Slides のスピーカーノートを読み上げた解説 MP4 を自動生成（`app/slides_narrate.py`）
> 4. **AI Runtime (AIR) ジョブ** — 推論 / LoRA ファインチューン / 量子化ベンチ（`air_job/*.yaml`）

## いちばん大事なこと

**「Databricks を全く知らない人でも、手順どおり実行すれば必ず同じようにサービングできる」** ことを目標に、
すべてのコマンドを [`docs/GUIDE.md`](docs/GUIDE.md) に**省略なし**で書いています。まずはそちらを読んでください。

## クイックスタート（サービングまでの最短経路）

```bash
# 0) 前提: Databricks CLI インストール済み・認証済み（詳細は GUIDE.md 第2章）
git clone <this-repo-url> && cd model-irodori-tts-on-databricks

# 1) 登録ノートブックをワークスペースへアップロード
databricks workspace import /Workspace/Users/<you>/tts_express/serve_irodori_tts_express \
  --file air_job/serving/serve_irodori_tts_express.py \
  --format SOURCE --language PYTHON --overwrite --profile DEFAULT

# 2) サーバレス GPU ジョブとして実行（自動でモデル取得→env_pack→UC登録, 約15-20分）
#    <catalog>.<schema> は書き込める場所に変更（GUIDE.md 参照）
databricks jobs submit --no-wait --profile DEFAULT --json '{
  "run_name": "irodori-tts-register",
  "tasks": [{"task_key":"register",
    "notebook_task": {"notebook_path":"/Workspace/Users/<you>/tts_express/serve_irodori_tts_express",
      "base_parameters": {"uc_model":"main.default.irodori_tts_express"}},
    "environment_key":"gpu_env","compute":{"hardware_accelerator":"GPU_1xA10"}}],
  "environments":[{"environment_key":"gpu_env","spec":{"client":"4"}}]}'

# 3) 登録できたら GPU サービングエンドポイントを作成（scale-to-zero）
databricks serving-endpoints create --no-wait --profile DEFAULT --json '{
  "name":"irodori-tts-express",
  "config":{"served_entities":[{"name":"m1","entity_name":"main.default.irodori_tts_express",
    "entity_version":"1","workload_type":"GPU_MEDIUM","workload_size":"Small","scale_to_zero_enabled":true}],
  "traffic_config":{"routes":[{"served_model_name":"m1","traffic_percentage":100}]}}}'

# 4) 問い合わせ（初回はコールドスタートで1-2分）
databricks serving-endpoints query irodori-tts-express --profile DEFAULT \
  --json '{"dataframe_records":[{"text":"こんにちは。音声合成のテストです。","params_json":"{}"}]}'
```

レスポンスの `audio_wav_b64` を base64 デコードすれば WAV が得られます（デコード例は GUIDE.md 第4章）。
全パラメータとレスポンスの仕様は [`docs/SERVING-API.md`](docs/SERVING-API.md)。

## リポジトリ構成

```
├── README.md                想像上の最短経路（このファイル）
├── docs/GUIDE.md            ★ 網羅的な手順書（未経験者向け・省略なし）
├── docs/SERVING-API.md      サービングエンドポイントの API 仕様（正本）
├── air_job/
│   ├── serving/serve_irodori_tts_express.py   サービング登録ノートブック（自己完結・本命）
│   ├── finetune/finetune_poc.py               LoRA ファインチューン疎通デモ
│   ├── bench_quant.py                          量子化・スループット計測
│   ├── airlog.py                               ジョブ用ロギングヘルパ
│   ├── databricks_text.txt                     サンプル入力テキスト
│   └── 01_infer_fp32.yaml / 03_finetune.yaml / 04_bench_quant.yaml   AIR CLI ジョブ定義
├── app/
│   ├── tts_app.py            ローカル再生アプリ（標準ライブラリのみ）
│   └── slides_narrate.py     Slides→ナレーション動画ジェネレータ
├── REPORT.md                 PoC の詳細レポート（設計判断・ハマりどころ）
└── NOTICE.md                 上流 Irodori-TTS (MIT) の謝辞・ライセンス
```

## ライセンス

モデル本体は上流 [Irodori-TTS](https://github.com/Aratako/Irodori-TTS)（MIT, Aratako 氏）。詳細は [NOTICE.md](NOTICE.md)。
