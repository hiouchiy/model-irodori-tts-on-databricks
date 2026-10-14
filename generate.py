#!/usr/bin/env python
"""Synthesize one Japanese utterance with Irodori-TTS (v4.1-Small or v4-Large; irodori_core.MODELS) to a WAV.

The entry point of the AI Runtime batch job (air_job/generate.yaml). Any machine with a CUDA GPU and the runtime
from scripts/install.sh can run it too. The request handling is irodori_core.Synthesizer, shared with the notebook
and the serving model, so `--params` takes any `params_json` key of the endpoint (docs/SERVING-API.md).

    python generate.py --text-file examples/databricks_text.txt --out /tmp/irodori_out/intro.wav \\
        --params '{"seed": 42, "max_text_len": 512}'
"""
import argparse
import base64
import json
from pathlib import Path

import irodori_core as core


def main() -> None:
    ap = argparse.ArgumentParser(description="Irodori-TTS synthesis (one utterance, ~30 s max)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Japanese text to read")
    src.add_argument("--text-file", help="Path to a UTF-8 text file to read")
    ap.add_argument("--out", required=True, help="Output .wav path (local disk; copy to a Volume afterwards)")
    ap.add_argument("--params", default="{}", help="JSON object of endpoint parameters (seed, caption, num_steps, ...)")
    ap.add_argument("--work", default="/tmp/irodori", help="Where the pinned upstream is cloned")
    ap.add_argument("--model", default=None, choices=sorted(core.MODELS),
                    help="checkpoint (default: $IRODORI_MODEL, else v4.1-small); README 'Choosing a model'")
    args = ap.parse_args()

    text = args.text if args.text is not None else Path(args.text_file).read_text(encoding="utf-8").strip()
    core.prepare(args.work)
    synth = core.Synthesizer(args.model)
    row = synth.handle(text, json.loads(args.params))
    if row["error"]:
        raise SystemExit(f"[irodori] synthesis failed: {row['error']}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(row["audio_wav_b64"]))
    out.with_suffix(".json").write_text(json.dumps({k: v for k, v in row.items() if k != "audio_wav_b64"},
                                                   ensure_ascii=False, indent=1), encoding="utf-8")

    import torch
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"[irodori] {row['audio_seconds']} s of audio in {row['synth_seconds']} s, peak GPU {peak:.2f} GiB -> {out}")


if __name__ == "__main__":
    main()
