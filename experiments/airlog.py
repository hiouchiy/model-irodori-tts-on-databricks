"""Lightweight logging / diagnostics helpers for AIR + Model Serving jobs.

Why this exists: when a job fails during environment setup (e.g. a `%pip` or
`uv sync` dependency conflict), the Jobs API often surfaces only an opaque
`CalledProcessError` — the actual pip resolver message is buried in cell output
that `jobs get-run-output` truncates. These helpers make every stage and every
subprocess self-logging, so the root cause is always visible in the run logs.

Usage (works in a plain script, an AIR `command:` python, or a notebook cell):

    from airlog import banner, stage, run, env_report, exc_guard

    env_report()                       # python / torch / cuda / GPU / key pkgs / pip constraints
    with stage("uv sync"):
        run(["uv", "sync", "--extra", "cu128"])   # streams + captures; prints tail on failure
    with exc_guard("log + register"):  # prints a full traceback then re-raises
        ...
"""
from __future__ import annotations

import contextlib
import glob
import subprocess
import sys
import time
import traceback


def banner(msg: str) -> None:
    """Print a timestamped, easy-to-grep stage banner."""
    print(f"\n[airlog {time.strftime('%H:%M:%S')}] ===== {msg} =====", flush=True)


@contextlib.contextmanager
def stage(name: str):
    """Time a stage and always print its outcome (OK / FAILED + elapsed)."""
    banner(f"BEGIN {name}")
    t0 = time.perf_counter()
    try:
        yield
    except BaseException as e:  # noqa: BLE001 — we re-raise; this only logs
        dt = time.perf_counter() - t0
        print(f"[airlog {time.strftime('%H:%M:%S')}] FAILED {name} after {dt:.1f}s: "
              f"{type(e).__name__}: {e}", flush=True)
        raise
    else:
        dt = time.perf_counter() - t0
        print(f"[airlog {time.strftime('%H:%M:%S')}] OK {name} ({dt:.1f}s)", flush=True)


@contextlib.contextmanager
def exc_guard(name: str):
    """Print a FULL traceback for any exception in the block, then re-raise.

    Use around log_model / register_model so serving-registration failures show
    their real stack in the run logs instead of a one-line message."""
    try:
        yield
    except BaseException:
        print(f"[airlog {time.strftime('%H:%M:%S')}] EXCEPTION in {name}:", flush=True)
        traceback.print_exc()
        raise


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess, ALWAYS printing stdout+stderr (tail on success, full-ish on failure).

    Prefer this over `%pip` / bare subprocess for dependency installs: the pip
    resolver message that explains a conflict is captured and printed here even
    when the Jobs API would otherwise hide it."""
    print(f"[airlog {time.strftime('%H:%M:%S')}] $ {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    out, err = p.stdout or "", p.stderr or ""
    if p.returncode == 0:
        if out:
            print(out[-1500:], flush=True)
    else:
        # Failure: print generous tails of BOTH streams so the real cause is visible.
        print(f"[airlog] !! exit={p.returncode}; STDOUT tail:\n{out[-3000:]}", flush=True)
        print(f"[airlog] !! STDERR tail:\n{err[-3000:]}", flush=True)
        if check:
            raise subprocess.CalledProcessError(p.returncode, cmd, out, err)
    return p


def pip_constraints() -> list[str]:
    """Return the paths of any Databricks serverless pip constraint files in effect.

    These `-c pipConstraints.txt` files silently pin package versions (e.g. torch)
    and are the usual reason a `%pip install torch==<newer>` fails on serverless."""
    return sorted(glob.glob("/local_disk0/.ephemeral_nfs/envs/*/pipConstraints.txt"))


def env_report(*, pkgs=("torch", "torchaudio", "torchcodec", "torchao", "transformers",
                        "mlflow", "peft", "safetensors", "numba", "soundfile")) -> None:
    """Print a one-shot environment banner: python, torch/CUDA, GPU, key pkg versions,
    and any active pip constraint on those packages."""
    banner("ENV REPORT")
    print("python:", sys.version.split()[0], flush=True)
    try:
        import torch  # noqa: PLC0415
        print("torch:", torch.__version__, "| cuda_avail:", torch.cuda.is_available(),
              "| torch.version.cuda:", torch.version.cuda, flush=True)
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            print(f"gpu: {torch.cuda.get_device_name(0)} cc={cc[0]}.{cc[1]}", flush=True)
    except Exception as e:  # noqa: BLE001
        print("torch: (not importable)", e, flush=True)

    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415
    for name in pkgs:
        try:
            print(f"  {name}: {version(name)}", flush=True)
        except PackageNotFoundError:
            print(f"  {name}: (not installed)", flush=True)

    for c in pip_constraints():
        try:
            lines = [ln for ln in open(c).read().splitlines()
                     if any(k in ln.lower() for k in pkgs)]
        except OSError:
            lines = []
        print(f"  pip constraint {c}: {len(lines)} relevant pins", flush=True)
        for ln in lines:
            print("     ", ln, flush=True)
