# Databricks notebook source
# MAGIC %md
# MAGIC # Irodori-TTS — register for Model Serving (GPU Express Deployment, env_pack)
# MAGIC
# MAGIC Run on **serverless GPU, `GPU_1xA10`, environment version 4** (a serverless client image is required: on classic
# MAGIC compute env_pack fails with `is_client_image=False`). Open it from a Git folder (or a synced copy) of this repo.
# MAGIC
# MAGIC Standard GPU Model Serving reinstalls torch 2.7.1+cu118 over the torch 2.10/cu128 this model needs
# MAGIC (`libcudart.so.12 not found` at load). **`env_pack="databricks_model_serving"`** instead packs THIS notebook's
# MAGIC environment and restores it at serving verbatim, so there is no container rebuild.
# MAGIC
# MAGIC 1. Install the pinned runtime (`scripts/install.sh`; the pins and why they are what they are: `irodori_core.py`).
# MAGIC 2. Clone the pinned upstream (`irodori_core.prepare`) and **smoke-test on the A10** with the exact serving PyFunc.
# MAGIC 3. Log the PyFunc (`serving/irodori_pyfunc.py`, code_paths = `irodori_tts` + `irodori_core.py`), then register it
# MAGIC    to Unity Catalog with env_pack and `install_dependencies=False`: git dependencies break env_pack's dependency
# MAGIC    step ("Cannot find child https:"), so it packs and restores without re-resolving.
# MAGIC
# MAGIC The HF checkpoint is not bundled (the upload got too big); the endpoint downloads it at load, so the serving
# MAGIC environment needs Hub egress.
# MAGIC
# MAGIC `model` picks the checkpoint (`irodori_core.MODELS`): `v4.1-small` (MIT) or `v4-large` (Gemma Terms of Use).
# MAGIC Register each as its own UC model and serve it from its own endpoint (README "Choosing a model").

# COMMAND ----------

import os
import subprocess
import sys

REPO = os.path.abspath("..")
subprocess.run(["bash", os.path.join(REPO, "scripts", "install.sh")], check=True,
               env={**os.environ, "PYTHON": sys.executable})
dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import shutil
import sys
import time

REPO = os.path.abspath("..")
sys.path[:0] = [REPO, os.path.join(REPO, "serving")]
import irodori_core as core  # noqa: E402

dbutils.widgets.dropdown("model", core.DEFAULT_MODEL, sorted(core.MODELS), "model (checkpoint)")
dbutils.widgets.text("uc_model", "<CATALOG>.<SCHEMA>.irodori_tts_express", "UC model (catalog.schema.name)")
MODEL = core.model_name(dbutils.widgets.get("model"))
UC_MODEL = dbutils.widgets.get("uc_model")
assert "<" not in UC_MODEL, "set uc_model to <catalog>.<schema>.<model>"
os.environ["IRODORI_MODEL"] = MODEL   # the smoke test's load_context(None) picks it up

WORK = "/tmp/irodori"
UPSTREAM = core.prepare(WORK)
import torch  # noqa: E402
core.log(f"torch {torch.__version__} cuda={torch.cuda.is_available()} upstream @ {core.UPSTREAM_SHA[:7]} model {MODEL}")

# COMMAND ----------

# MAGIC %md ## Smoke test with the exact serving PyFunc on this A10

# COMMAND ----------

import pandas as pd  # noqa: E402
from irodori_pyfunc import IrodoriTTSModel  # noqa: E402

m = IrodoriTTSModel()
m.load_context(None)
row = m.predict(None, pd.DataFrame([{"text": "こんにちは。これはデータブリックスのモデルサービング上で動く音声合成のテストです。",
                                     "params_json": json.dumps({"seed": 42})}])).to_dict(orient="records")[0]
assert not row["error"], row["error"]
SMOKE = {"model": MODEL, "audio_seconds": row["audio_seconds"], "synth_seconds": row["synth_seconds"],
         "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
core.log(f"smoke ok: {SMOKE}")
del m; torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md ## Log, then register with env_pack (install_dependencies=False)

# COMMAND ----------

import mlflow  # noqa: E402
from mlflow.utils.env_pack import EnvPackConfig  # noqa: E402

mlflow.set_registry_uri("databricks-uc")
code_dir = f"{WORK}/code"
shutil.rmtree(code_dir, ignore_errors=True)
shutil.copytree(UPSTREAM / "irodori_tts", f"{code_dir}/irodori_tts", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy(os.path.join(REPO, "irodori_core.py"), code_dir)
pyfunc = f"{WORK}/irodori_pyfunc.py"
shutil.copy(os.path.join(REPO, "serving", "irodori_pyfunc.py"), pyfunc)

# Two stable string columns: `text` + `params_json` (opaque JSON of any SamplingRequest field).
input_example = pd.DataFrame([{"text": "こんにちは、これはテストです。",
                               "params_json": json.dumps({"num_steps": 40, "cfg_scale_text": 3.0})}])
t0 = time.time()
info = mlflow.pyfunc.log_model(
    name="irodori_tts",
    python_model=pyfunc,
    code_paths=[f"{code_dir}/irodori_tts", f"{code_dir}/irodori_core.py"],
    model_config={"model": MODEL, "hf_checkpoint": core.MODELS[MODEL]["repo"], "revision": core.MODELS[MODEL]["revision"],
                  "precision": "bf16", "upstream_sha": core.UPSTREAM_SHA},
    input_example=input_example,
    extra_pip_requirements=core.SERVING_PIP_REQUIREMENTS,   # no URLs / git: keeps env_pack happy
)
core.log(f"logged {info.model_uri}")
mv = mlflow.register_model(model_uri=info.model_uri, name=UC_MODEL,
                           env_pack=EnvPackConfig(name="databricks_model_serving", install_dependencies=False))
core.log(f"registered {UC_MODEL} version {mv.version} in {time.time() - t0:.0f}s")

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"uc_model": UC_MODEL, "model": MODEL, "version": mv.version, "smoke": SMOKE}))
