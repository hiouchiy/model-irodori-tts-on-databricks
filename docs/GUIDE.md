# Irodori-TTS on Databricks — 完全手順書

> **この手順書のゴール**
> Databricks を **まったく触ったことがない人**でも、上から順にコマンドをコピー&ペーストして実行すれば、
> **日本語音声合成モデル Irodori-TTS を Databricks 上でサービング（GPU エンドポイント化）でき、
> テキストから音声を生成できる**ようになること。
>
> 迷ったら「章の順番どおり」に進めてください。第2〜4章までやれば、サービングは完成します。
> 第5章以降（アプリ・スライド動画・AIR ジョブ）は応用です。

---

## 目次
0. [全体像と仕組み](#0-全体像と仕組み)
1. [前提条件（アカウント・権限・ローカル環境）](#1-前提条件)
2. [Databricks CLI のインストールと認証](#2-databricks-cli-のインストールと認証)
3. [このリポジトリを取得](#3-このリポジトリを取得)
4. [★モデルをサービングする（本命・最短経路）](#4-モデルをサービングする本命)
5. [サービング API 仕様（パラメータ一覧）](#5-サービング-api-仕様)
6. [ローカル再生アプリ](#6-ローカル再生アプリ)
7. [スライド → ナレーション動画](#7-スライド--ナレーション動画)
8. [AI Runtime (AIR) ジョブ：推論・ファインチューン・ベンチ](#8-ai-runtime-air-ジョブ)
9. [トラブルシューティング](#9-トラブルシューティング)
10. [コストと後片付け](#10-コストと後片付け)

---

## 0. 全体像と仕組み

**Irodori-TTS** は Aratako 氏による日本語音声合成の OSS（MIT ライセンス）です。torch 2.10 / CUDA 12.8
という**最先端スタック**を要求するため、Databricks の標準 GPU Model Serving（torch を自前バージョンに固定する）
では**そのままでは載りません**。

そこで本手順では **env_pack（GPU Express Deployment）** を使います。これは
**「サーバレス GPU ノートブックで作った Python 環境をそのままパッケージ化し、サービング時に復元する」**
仕組みで、コンテナの再ビルドをしないため torch の上書きが起きません。

処理の流れ:

```
[登録ノートブック(サーバレスGPU)]
   1. 上流 Irodori-TTS を git clone
   2. torch 2.10/cu128 等を2フェーズでpip install（依存の地雷を回避）
   3. Hugging Face からモデルを取得しスモーク合成
   4. pyfunc として MLflow に log
   5. env_pack 付きで Unity Catalog にモデル登録
        ↓
[Model Serving]
   6. 登録したモデルから GPU エンドポイントを作成（scale-to-zero）
   7. REST でテキストを送ると base64 の WAV が返る
```

ハマりどころ（protobuf 競合・torchvision の ABI 不整合・大容量アップロードのタイムアウト）は
**登録ノートブックが自動で対処済み**です。詳しくは [../REPORT.md](../REPORT.md) を参照。

---

## 1. 前提条件

### 1-1. Databricks 側
- **Databricks ワークスペース**（AWS / Azure / GCP いずれも可）。
- **Unity Catalog** が有効で、モデルを登録できる **カタログ.スキーマ** があること
  （書き込み権限 = スキーマに対する `CREATE MODEL`、`USE CATALOG`/`USE SCHEMA`）。
  多くのワークスペースにある `main.default` が使えれば最短です。
- **サーバレス GPU**（Serverless GPU compute）が利用可能なこと（登録ノートブックの実行に必要）。
- **Model Serving（GPU）** が利用可能なこと（`GPU_MEDIUM` = NVIDIA A10 を使用）。
- （第7章のスライド機能で LLM 分割を使う場合のみ）**Foundation Model API** が有効なこと。

> リージョンやプランによりサーバレス GPU / GPU Serving が使えないことがあります。管理者にご確認ください。

### 1-2. ローカル（手元の PC）側
- **Python 3.10 以上**（`python3 --version`）。
- **Databricks CLI**（第2章でインストール）。
- **ffmpeg**（第7章の動画生成にのみ必要。`ffmpeg -version`）。
- インターネット接続。

### 1-3. コスト感
- 登録ジョブ: サーバレス GPU（A10）1台を約 15〜20 分。
- サービング: GPU_MEDIUM（A10）。**scale-to-zero** を有効にするので、リクエストが無い間は課金が抑えられます
  （初回リクエストはコールドスタートで 1〜2 分かかります）。
- 使い終わったらエンドポイントを削除（第10章）。

---

## 2. Databricks CLI のインストールと認証

### 2-1. インストール

**macOS (Homebrew):**
```bash
brew tap databricks/tap
brew install databricks
```

**Linux / macOS (共通スクリプト):**
```bash
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
```

確認:
```bash
databricks version   # 例: Databricks CLI v0.x.x
```

### 2-2. 認証（プロファイル作成）

ブラウザで OAuth 認証します。`<workspace-url>` はご自身のワークスペース URL
（例: `https://dbc-xxxx.cloud.databricks.com`、Azure なら `https://adb-xxxx.azuredatabricks.net`）。

```bash
databricks auth login --host https://<workspace-url> --profile DEFAULT
```

ブラウザが開くのでログイン→承認。これで `~/.databrickscfg` に `DEFAULT` プロファイルが作られます。

> 以降のコマンドはすべて `--profile DEFAULT` を付けています。別名にした場合は読み替えてください。

### 2-3. 認証確認

```bash
databricks current-user me --profile DEFAULT        # 自分のユーザー情報が出れば OK
databricks catalogs list --profile DEFAULT          # 書き込めるカタログを確認
```

`current-user me` の出力の `userName`（例 `you@example.com`）を控えておきます。後の
ワークスペースパス `/Workspace/Users/<userName>/...` で使います。

---

## 3. このリポジトリを取得

```bash
git clone <this-repo-url>
cd irodori-tts-databricks
```

> 上流の Irodori-TTS 本体は**同梱していません**。サービング用ノートブックが実行時に自動で
> `git clone` します（第4章）。手元でクローンするのは第8章（AIR ジョブ）を行う場合だけです。

---

## 4. モデルをサービングする（本命）

ここが本手順の中心です。**4-1 → 4-6 を順番に**実行してください。

以下では次のプレースホルダを使います。自分の値に置き換えてください:
- `<userName>` … 2-3 で確認したメールアドレス形式のユーザー名
- `<catalog>.<schema>` … モデルを登録する場所（例 `main.default`）
- `<endpoint>` … 作成するエンドポイント名（例 `irodori-tts-express`）

### 4-1. 登録ノートブックをワークスペースへアップロード

```bash
databricks workspace import \
  /Workspace/Users/<userName>/tts_express/serve_irodori_tts_express \
  --file air_job/serving/serve_irodori_tts_express.py \
  --format SOURCE --language PYTHON --overwrite --profile DEFAULT
```

> `--format SOURCE --language PYTHON` により、`# COMMAND ----------` 区切りの Python ファイルが
> Databricks ノートブックとして取り込まれます。

### 4-2. 登録ジョブを実行（サーバレス GPU）

`<catalog>.<schema>` を書き込める場所に変えて実行します。

```bash
databricks jobs submit --no-wait --profile DEFAULT --json '{
  "run_name": "irodori-tts-register",
  "tasks": [{
    "task_key": "register",
    "notebook_task": {
      "notebook_path": "/Workspace/Users/<userName>/tts_express/serve_irodori_tts_express",
      "base_parameters": {"uc_model": "<catalog>.<schema>.irodori_tts_express"}
    },
    "environment_key": "gpu_env",
    "compute": {"hardware_accelerator": "GPU_1xA10"}
  }],
  "environments": [{"environment_key": "gpu_env", "spec": {"client": "4"}}]
}'
```

- `compute.hardware_accelerator: GPU_1xA10` … サーバレス GPU（A10 ×1）。
- `environments[].spec.client: "4"` … **サーバレス GPU の client image。これが無いと失敗します**。
- `base_parameters.uc_model` … 登録先。ノートブック側のウィジェット `uc_model` に渡されます。

実行すると `run_id`（数字）が返ります。**この番号を控えます**（`<RUN_ID>`）。

> このジョブが内部でやること（全自動）: 上流クローン → torch 2.10/cu128 等の 2 フェーズ install
> → torchvision の一致版導入 → Hugging Face からモデル取得＆スモーク合成 → pyfunc を log →
> env_pack 付きで UC 登録。**約 15〜20 分**かかります。

### 4-3. ジョブの進捗・結果を確認

```bash
# 状態（RUNNING / TERMINATED など）
databricks jobs get-run <RUN_ID> --profile DEFAULT | \
  python3 -c "import json,sys;s=json.load(sys.stdin)['state'];print(s.get('life_cycle_state'),s.get('result_state',''))"

# 完了後、ノートブックの戻り値（登録されたモデルとバージョン）を取得
TASK_RUN_ID=$(databricks jobs get-run <RUN_ID> --profile DEFAULT | python3 -c "import json,sys;print(json.load(sys.stdin)['tasks'][0]['run_id'])")
databricks jobs get-run-output $TASK_RUN_ID --profile DEFAULT | \
  python3 -c "import json,sys;print(json.load(sys.stdin).get('notebook_output',{}).get('result'))"
```

成功すると `{"uc_model": "<catalog>.<schema>.irodori_tts_express", "version": "1"}` のように出ます。
**この `version`（例 1）を控えます**（`<VERSION>`）。

> 失敗した場合は同じ `get-run-output` の `error` にトレースバックが出ます。第9章参照。

### 4-4. GPU サービングエンドポイントを作成

`<VERSION>` は 4-3 で得たバージョン番号にします。

```bash
databricks serving-endpoints create --no-wait --profile DEFAULT --json '{
  "name": "<endpoint>",
  "config": {
    "served_entities": [{
      "name": "irodori_tts_express-<VERSION>",
      "entity_name": "<catalog>.<schema>.irodori_tts_express",
      "entity_version": "<VERSION>",
      "workload_type": "GPU_MEDIUM",
      "workload_size": "Small",
      "scale_to_zero_enabled": true
    }],
    "traffic_config": {"routes": [{"served_model_name": "irodori_tts_express-<VERSION>", "traffic_percentage": 100}]}
  }
}'
```

- `workload_type: GPU_MEDIUM` … A10。**登録時と同じ GPU 系に合わせます**（env_pack は環境をそのまま復元するため）。
- `scale_to_zero_enabled: true` … 無リクエスト時にゼロ台へ縮退（待機課金を抑制）。

### 4-5. READY になるまで待つ

```bash
databricks serving-endpoints get <endpoint> --profile DEFAULT | \
  python3 -c "import json,sys;s=json.load(sys.stdin)['state'];print('ready=',s.get('ready'),' update=',s.get('config_update'))"
```

`ready=READY update=NOT_UPDATING` になれば完成です（初回のイメージ復元に数分かかります）。

### 4-6. テキストから音声を生成（問い合わせ）

入力は **2 つの文字列カラム** `text` と `params_json` です（`params_json` は空 `{}` でも必ず含めます）。

```bash
databricks serving-endpoints query <endpoint> --profile DEFAULT \
  --json '{"dataframe_records":[{"text":"こんにちは。これはデータブリックス上の音声合成のテストです。","params_json":"{}"}]}' \
  > /tmp/tts_resp.json
```

> **初回はコールドスタートで 1〜2 分**かかります。CLI の `query` には約 60 秒の内部タイムアウトがあるため、
> 初回はタイムアウトすることがあります。その場合は数十秒おいて再実行するか、下の Python（600 秒待ち）を使ってください。

返ってきた `audio_wav_b64` を WAV に保存する例:

```bash
python3 - <<'PY'
import json, base64
d = json.load(open("/tmp/tts_resp.json"))
rec = (d.get("predictions") or d)[0]
open("out.wav","wb").write(base64.b64decode(rec["audio_wav_b64"]))
print("saved out.wav :", rec["sample_rate"], "Hz,", rec["audio_seconds"], "s (synth", rec["synth_seconds"], "s)")
PY
```

コールドスタートに強い **REST 直叩き**（Python, 600 秒待ち）例:

```bash
python3 - <<'PY'
import json, base64, subprocess, urllib.request, configparser, os
PROFILE, ENDPOINT = "DEFAULT", "<endpoint>"
cfg = configparser.ConfigParser(); cfg.read(os.path.expanduser("~/.databrickscfg"))
host = cfg[PROFILE]["host"].rstrip("/")
token = json.loads(subprocess.check_output(["databricks","auth","token","-p",PROFILE]))["access_token"]
body = json.dumps({"dataframe_records":[{"text":"長文でもOKです。","params_json":"{}"}]}).encode()
req = urllib.request.Request(f"{host}/serving-endpoints/{ENDPOINT}/invocations",
    data=body, headers={"Authorization":f"Bearer {token}","Content-Type":"application/json"})
rec = json.loads(urllib.request.urlopen(req, timeout=600).read())["predictions"][0]
open("out.wav","wb").write(base64.b64decode(rec["audio_wav_b64"]))
print("saved out.wav", rec["audio_seconds"], "s")
PY
```

**これでサービング完成です。** 以降は応用機能です。

---

## 5. サービング API 仕様

リクエストの全カラム・`params_json` の全キー・レスポンス・例・性能・制約は **[SERVING-API.md](SERVING-API.md)** にまとめてある（仕様の正本。ここには写さない）。

---

## 6. ローカル再生アプリ

ブラウザからテキストを入れて音声を生成・再生できる、依存ゼロ（Python 標準ライブラリのみ）のアプリです。
認証は Databricks CLI プロファイルを流用します。

```bash
python3 app/tts_app.py --profile DEFAULT --endpoint <endpoint>
# → http://127.0.0.1:8765 をブラウザで開く（自動で開かない場合は --no-open を外すか手動で）
```

機能:
- **長文分割**（1 回 ~24 秒の塊にまとめて連結）: 「セマンティック分割 (LLM)」または「文字数ベース」。
  セマンティックは Foundation Model API（既定 `databricks-claude-haiku-4-5`）で意味の切れ目に分割します。
- **声を固定**: 最初に参照音声を 1 つ作り、全チャンクをそれに揃える（連結しても声がぶれない）。
- **主要パラメータ UI** ＋「追加パラメータ (JSON)」で任意の `params_json` を指定。

> 長文の継ぎ目ノイズ対策として、チャンク境界に数 ms のフェードを掛けています。

---

## 7. スライド → ナレーション動画

Google Slides の各スライドの**スピーカーノート**を読み上げ、スライド画像と合成して
**1 本の解説 MP4** を自動生成します（`app/slides_narrate.py`）。

### 7-1. 追加の前提
- **ffmpeg** がインストール済み。
- **Google OAuth アクセストークン**（スコープ: Slides 読み取り = `presentations`、Drive = `drive`）。
  `token` サブコマンドでアクセストークンを標準出力するヘルパスクリプトを用意し、そのパスを
  環境変数 `GOOGLE_AUTH_SCRIPT` で指定します。

```bash
export GOOGLE_AUTH_SCRIPT=/path/to/google_auth.py   # `python google_auth.py token` がトークンを出力する想定
```

> 社内 Vibe 環境では `~/.vibe/marketplace/plugins/fe-google-tools/skills/google-auth/resources/google_auth.py`
> が既定パスです（未設定時のフォールバック）。自前で用意する場合は「`token` を渡すとアクセストークンを
> 1 行で出力する」実装にしてください。`app/slides_narrate.py` 冒頭の `QUOTA`（GCP クォータプロジェクト）も
> 環境に合わせて調整します。

### 7-2. 実行

```bash
# 本番デッキをコピー(--copy)して、ノートを校正(--clean)し、先頭60ページをMP4化
python3 app/slides_narrate.py \
  --url "https://docs.google.com/presentation/d/<PRESENTATION_ID>/edit" \
  --copy --clean --pages 60 \
  --out ./outputs/deck60.mp4
```

- `--copy` … 本番デッキを Drive で複製（**原本は無変更**）。以降は複製に対して処理。
- `--clean` … 各ノートを LLM（既定 `databricks-claude-sonnet-4-5`）で校正:
  **難読漢字・英単語・製品名・略語をカタカナ読みに置換**し、古い/重複した下書きを削除。
  校正後テキストは複製のノートに書き戻し、`notes_cleaned.json` にも保存。
  校正で内容が変わったスライドだけ音声を作り直します（差分再合成）。
- `--pages N` … 先頭 N ページのみ（`0` で全ページ）。
- 途中でエンドポイント呼び出しが失敗しても**自動リトライ**、`slideNN.wav` が既にあれば**再開時に再利用**。
- 参照音声は `_ref.wav` にキャッシュし、全ページで同じナレーター声に統一。

> `--clean` は無指定でも動きます（校正なしでノートをそのまま読み上げ）。校正は「TTS が英字や難読漢字を
> 誤読しやすい」ための保険です。強度が気になる場合は `slides_narrate.py` の `clean_note` プロンプトを調整。

---

## 8. AI Runtime (AIR) ジョブ

推論・LoRA ファインチューン・量子化ベンチを、AIR CLI ジョブとして実行する例です（サービングとは別系統）。
これらは**上流リポジトリの中で** `uv`（`uv.lock` 由来の torch 2.10/cu128 スタック）で動きます。

### 8-1. 準備
```bash
# 上流を「動作確認済みコミット」で取得
git clone https://github.com/Aratako/Irodori-TTS.git
cd Irodori-TTS
git checkout 89f9d8fbd4d51ea019867ee1197725ede1df13c5
# 本リポジトリの air_job/ を上流へ配置
cp -r ../irodori-tts-databricks/air_job ./air_job
```

### 8-2. Volume パスを自分の環境に変更
`air_job/01_infer_fp32.yaml` / `03_finetune.yaml` / `04_bench_quant.yaml` の中の
`/Volumes/<catalog>/<schema>/tts/...` を、自分が書き込める **Unity Catalog Volume** のパスに変更します
（Volume が無ければ `databricks volumes create` で作成、または UI から作成）。

### 8-3. 実行（AIR CLI）
```bash
# 例: FP32 推論（GPU A10, 約数分）。CLI の air サブコマンドでジョブ投入
databricks air submit --profile DEFAULT --config air_job/01_infer_fp32.yaml
```
> ジョブ内で `uv sync --extra cu128` により torch 2.10/cu128 等をロックファイルから再現します。
> 生成物は YAML 内の Volume パスに保存されます。`03_finetune.yaml`（LoRA 疎通）、`04_bench_quant.yaml`
> （量子化・スループット計測）も同様です。詳細な結果は [../REPORT.md](../REPORT.md) を参照。

---

## 9. トラブルシューティング

| 症状 | 原因 / 対処 |
|---|---|
| 登録ジョブが `client image` 関連で失敗 | `environments[].spec.client: "4"` が抜けている / サーバレス GPU 未有効。4-2 の JSON を確認 |
| `serving-endpoints query` が 60 秒で失敗 | エンドポイントのコールドスタート。数十秒後に再実行、または 4-6 の REST（600 秒）版を使用 |
| 登録が `UC` 権限エラー | `uc_model` のスキーマに `CREATE MODEL` 権限が無い。書き込めるカタログ.スキーマに変更 |
| エンドポイント作成が `GPU` 型で失敗 | ワークスペースで GPU Serving が未提供 / クォータ不足。管理者に確認 |
| 生成音声が 30 秒で切れる/崩れる | 1 回の合成上限（~256 トークン / ~30 秒）。長文は第6章のアプリ（分割連結）を使う |
| 英字や固有名詞を誤読する | 第7章 `--clean` のようにカタカナへ置換、または `params_json` を工夫 |
| ノートブックの `print` が見えない | サーバレスの `print`/`%pip` 出力は `get-run-output` に出ない。戻り値は `dbutils.notebook.exit` の JSON で確認 |

登録ジョブのエラー詳細は `databricks jobs get-run-output <TASK_RUN_ID>` の `error` に出ます。
既知の依存問題（protobuf / torchvision / アップロードタイムアウト）は登録ノートブックが自動対処済みです
（背景は [../REPORT.md](../REPORT.md)）。

---

## 10. コストと後片付け

```bash
# エンドポイント一覧
databricks serving-endpoints list --profile DEFAULT | grep irodori || true

# 使い終わったらエンドポイント削除（GPU 待機課金を止める）
databricks serving-endpoints delete <endpoint> --profile DEFAULT
```

- `scale_to_zero_enabled: true` なら無リクエスト時は自動縮退しますが、完全に止めるには削除します。
- UC のモデル（`<catalog>.<schema>.irodori_tts_express`）はストレージのみで課金は僅少。不要なら UI から削除可。

---

### 関連ドキュメント
- [../README.md](../README.md) … 概要・最短経路
- [../REPORT.md](../REPORT.md) … PoC の詳細（設計判断・ハマりどころ・ベンチ結果）
- [../NOTICE.md](../NOTICE.md) … 上流 Irodori-TTS (MIT) の謝辞・ライセンス
