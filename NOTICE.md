# NOTICE

This repo is a **thin wrapper, procedures and tools** to run and serve **Irodori-TTS** (by Aratako, MIT License) on
Databricks. It does not include the model's code: the notebooks and the AIR job clone the upstream repo at a pinned
commit at run time, and the weights are downloaded from the Hugging Face Hub.

- Upstream: https://github.com/Aratako/Irodori-TTS
- Licence: MIT License (Copyright (c) 2026 Aratako)
- Pinned commit: `89f9d8fbd4d51ea019867ee1197725ede1df13c5` (2026-09-12; `irodori_core.UPSTREAM_SHA`)
- Model (Hugging Face): `Aratako/Irodori-TTS-v4.1-Small`

Check the upstream licence terms yourself before use. Code written in this repo is provided under the MIT License, the
same as upstream.
