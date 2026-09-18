# NOTICE / 謝辞・ライセンス

このリポジトリは **Irodori-TTS**（作者: Aratako 氏, MIT License）を Databricks 上で
動かす／サービングするための **薄いラッパー・手順・ツール** です。モデル本体のコードは
同梱していません。サービング用ノートブックが実行時に上流リポジトリを取得します。

- 上流リポジトリ: https://github.com/Aratako/Irodori-TTS
- ライセンス: MIT License (Copyright (c) 2026 Aratako)
- 動作確認したコミット（再現性のためのピン）: `89f9d8fbd4d51ea019867ee1197725ede1df13c5` (2026-09-12)
- モデル（Hugging Face）: `Aratako/Irodori-TTS-v4.1-Small`

上流のライセンス条項は各自ご確認ください。本リポジトリ内で新規に作成したコード
（`air_job/`, `app/`, `docs/`）は、上流の MIT ライセンスに準じて扱ってください。
