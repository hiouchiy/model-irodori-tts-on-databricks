# PoC AIR jobs (not one of the three ways to run)

The AI Runtime jobs from the PoC ([docs/REPORT.md](../docs/REPORT.md)). Unlike the repo's ways to run, they run
**inside an upstream checkout** and reproduce its environment with `uv sync` from upstream's `uv.lock`.

| Job | What it does | Report |
|---|---|---|
| `01_infer_fp32.yaml` | fp32 inference with upstream's `infer.py` | §1 |
| `03_finetune.yaml` → `finetune/finetune_poc.py` | 30-step LoRA plumbing check: generate clips → encode → train → reload | §3 |
| `04_bench_quant.yaml` → `bench_quant.py` (+ `airlog.py`) | fp32 / bf16 / int8 / int4 / float8 speed and VRAM | §4 |

```bash
git clone https://github.com/Aratako/Irodori-TTS.git && cd Irodori-TTS
git checkout 89f9d8fbd4d51ea019867ee1197725ede1df13c5          # irodori_core.UPSTREAM_SHA
cp -r ../model-irodori-tts-on-databricks/experiments ./air_job   # the jobs expect <upstream>/air_job/
cp ../model-irodori-tts-on-databricks/examples/databricks_text.txt ./air_job/
# set the /Volumes/<CATALOG>/<SCHEMA>/<VOLUME> paths in the YAML, then:
COPYFILE_DISABLE=1 air run --file air_job/01_infer_fp32.yaml --watch --profile <PROFILE>
```
