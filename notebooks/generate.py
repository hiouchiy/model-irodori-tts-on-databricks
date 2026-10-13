# Databricks notebook source
# MAGIC %md
# MAGIC # Irodori-TTS — synthesize Japanese speech on serverless GPU
# MAGIC
# MAGIC Attach this notebook to **serverless GPU compute, accelerator A10** (environment version 4). Open it from a
# MAGIC Git folder (or a synced copy) of this repo, so `../irodori_core.py` and `../scripts/install.sh` are next to it.
# MAGIC
# MAGIC 1. Fill in the widgets. `params_json` takes **any** parameter of the endpoint (docs/SERVING-API.md):
# MAGIC    seed, caption, num_steps, duration_scale, cfg_scale_text, ref_wav_b64, ...
# MAGIC 2. Run all. The first run installs the runtime (torch 2.10/cu128, ~5 min) and downloads the checkpoint;
# MAGIC    an utterance then takes a few seconds.
# MAGIC
# MAGIC One utterance per run (~256 tokens / ~30 s). The request handling is the serving endpoint's
# MAGIC (`irodori_core.Synthesizer`).

# COMMAND ----------

dbutils.widgets.text("text", "こんにちは。これはデータブリックスのサーバーレス GPU で動く音声合成のテストです。", "text")
dbutils.widgets.text("params_json", '{"seed": 42}', "params_json")
dbutils.widgets.text("output_dir", "/Volumes/<CATALOG>/<SCHEMA>/<VOLUME>/irodori", "output_dir (UC Volume)")
dbutils.widgets.text("work_dir", "/tmp/irodori", "work_dir (upstream clone)")

# COMMAND ----------

# Install the runtime into THIS notebook's Python (same recipe as the AIR job), then restart Python.
import os
import subprocess
import sys

REPO = os.path.abspath("..")
subprocess.run(["bash", os.path.join(REPO, "scripts", "install.sh")], check=True,
               env={**os.environ, "PYTHON": sys.executable})
dbutils.library.restartPython()

# COMMAND ----------

import base64
import json
import os
import sys
import time

REPO = os.path.abspath("..")
sys.path.insert(0, REPO)
import irodori_core as core  # noqa: E402

out_dir = dbutils.widgets.get("output_dir")
assert "<" not in out_dir, "set output_dir to a UC Volume folder, e.g. /Volumes/main/default/tts/irodori"

core.prepare(dbutils.widgets.get("work_dir"))
t0 = time.time()
synth = core.Synthesizer()
core.log(f"runtime loaded in {time.time() - t0:.0f}s")
row = synth.handle(dbutils.widgets.get("text"), dbutils.widgets.get("params_json"))
assert not row["error"], row["error"]
core.log(f"{row['audio_seconds']} s of audio in {row['synth_seconds']} s (seed {row['used_seed']})")

# COMMAND ----------

os.makedirs(out_dir, exist_ok=True)
name = f"{int(time.time())}.wav"
data = base64.b64decode(row["audio_wav_b64"])
dest = os.path.join(out_dir, name)
open(dest, "wb").write(data)   # sequential write; libsndfile's seeking writer fails on Volume paths
open(dest + ".json", "w").write(json.dumps({k: v for k, v in row.items() if k != "audio_wav_b64"},
                                           ensure_ascii=False, indent=1))
print("wrote", dest)

from IPython.display import Audio, display  # noqa: E402
display(Audio(data=data))

# COMMAND ----------

import torch  # noqa: E402
dbutils.notebook.exit(json.dumps({"file": dest, "audio_seconds": row["audio_seconds"], "synth_seconds": row["synth_seconds"],
                                  "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}))
