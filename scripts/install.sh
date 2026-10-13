#!/bin/bash
# Install the Irodori-TTS runtime on serverless GPU (environment version 4). torch 2.10 / cu128 replaces the
# preinstalled torch. Used by the notebooks and air_job/generate.yaml; the pins live in irodori_core.py.
# The serving registration (notebooks/register_serving.py) packs the environment this leaves (env_pack).
#
#   bash scripts/install.sh                         # from the repo root (AIR job)
#   PYTHON=/path/to/python bash scripts/install.sh  # into a specific interpreter (the notebooks pass their own)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export HF_HUB_ENABLE_HF_TRANSFER=0     # serverless GPU presets it to 1 but does not install hf_transfer
INDEX=$("$PY" -c "import irodori_core; print(irodori_core.TORCH_INDEX)")
mapfile -t REQS < <("$PY" -c "import irodori_core; print('\n'.join(irodori_core.RUNTIME_REQUIREMENTS))")
mapfile -t NODEPS < <("$PY" -c "import irodori_core; print('\n'.join(irodori_core.NO_DEPS_REQUIREMENTS))")
# Phase 1: the runtime. Phase 2: packages whose stale protobuf<3.20 pin must be ignored (see irodori_core.py).
"$PY" -m pip install -q --no-cache-dir --extra-index-url "$INDEX" "${REQS[@]}" 2>&1 | tail -30
"$PY" -m pip install -q --no-cache-dir --no-deps "${NODEPS[@]}" 2>&1 | tail -10
"$PY" -m pip check || true             # reports descript-audiotools' protobuf pin; expected
"$PY" -c "import torch, torchvision; print('[install] torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda', torch.cuda.is_available())"
