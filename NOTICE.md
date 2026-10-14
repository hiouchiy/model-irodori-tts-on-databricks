# NOTICE

This repo is a **thin wrapper, procedures and tools** to run and serve **Irodori-TTS** (by Aratako; code MIT) on
Databricks. It does not include the model's code: the notebooks and the AIR job clone the upstream repo at a pinned
commit at run time, and the weights are downloaded from the Hugging Face Hub.

- Upstream: https://github.com/Aratako/Irodori-TTS
- Licence: MIT License (Copyright (c) 2026 Aratako)
- Pinned commit: `89f9d8fbd4d51ea019867ee1197725ede1df13c5` (2026-09-12; `irodori_core.UPSTREAM_SHA`)
- Models (Hugging Face), revisions pinned in `irodori_core.MODELS`:
  - `Aratako/Irodori-TTS-v4.1-Small` @ `2b28324dc263ed5e6638b3cf3dd94c82ead07b4b`, MIT License
  - `Aratako/Irodori-TTS-v4-Large` @ `2e0c55428ce97268a507f1feeb2478f8d9148e8b`, under the **Gemma Terms of Use**: its
    text/caption encoder is derived from `google/t5gemma-2-1b-1b`. The model's NOTICE reads: "Gemma is provided under
    and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms". Its use is also subject to the
    [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) and to the model card's ethical
    restrictions (no impersonation without consent, no deceptive deepfakes).

Check the upstream licence terms yourself before use. Code written in this repo is provided under the MIT License, the
same as upstream.
