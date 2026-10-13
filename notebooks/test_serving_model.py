# Databricks notebook source
# MAGIC %md
# MAGIC # Irodori-TTS — test the serving model before registering it
# MAGIC
# MAGIC Runs `serving/irodori_pyfunc.py` (the MLflow PyFunc the endpoint serves) in-process on **serverless GPU, A10**:
# MAGIC `load_context` + one `predict` with three rows (synthesis, `__list_params__`, an
# MAGIC unreadable reference WAV), and checks the
# MAGIC response shape. Use it after changing `serving/irodori_pyfunc.py` or `irodori_core.py`, before
# MAGIC `notebooks/register_serving.py`. It does not touch any endpoint.

# COMMAND ----------

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

import pandas as pd

REPO = os.path.abspath("..")
sys.path[:0] = [REPO, os.path.join(REPO, "serving")]
import irodori_core as core  # noqa: E402
core.prepare("/tmp/irodori")
from irodori_pyfunc import IrodoriTTSModel  # noqa: E402

m = IrodoriTTSModel()
m.load_context(None)
req = pd.DataFrame([{"text": "こんにちは。サービング用モデルのテストです。", "params_json": json.dumps({"seed": 42})},
                    {"text": "", "params_json": json.dumps({"__list_params__": True})},
                    {"text": "テスト", "params_json": json.dumps({"ref_wav_b64": base64.b64encode(b"not a wav").decode()})}])
ok, lp, bad = m.predict(None, req).to_dict(orient="records")
assert ok["error"] == "" and ok["sample_rate"] == 48000 and len(base64.b64decode(ok["audio_wav_b64"])) > 10_000, ok["error"]
assert "num_steps" in json.loads(lp["messages"])["supported_params"], lp
assert bad["error"] and bad["audio_wav_b64"] == "", bad
print(json.dumps({k: v for k, v in ok.items() if k != "audio_wav_b64"}, ensure_ascii=False))

# COMMAND ----------

import torch  # noqa: E402
dbutils.notebook.exit(json.dumps({"synth_ok": True, "audio_seconds": ok["audio_seconds"], "synth_seconds": ok["synth_seconds"],
                                  "list_params_ok": True, "bad_param_error": bad["error"][:200],
                                  "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}))
