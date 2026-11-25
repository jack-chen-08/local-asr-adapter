#!/usr/bin/env python3
"""Fetch the FunASR Paraformer model into a local directory (build-time step).

Run once at image build time; the runtime container then works with --network none.
Prints the resolved model dir and its size so the operator can record it as evidence.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time


def dir_size(path: pathlib.Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> int:
    model = os.getenv("IP_ASR_MODEL", "paraformer-zh")
    cache_dir = pathlib.Path(os.getenv("IP_ASR_MODEL_DIR", "models")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] model={model} cache_dir={cache_dir}", flush=True)

    from funasr import AutoModel  # imported here so --help style runs stay cheap

    started = time.monotonic()
    AutoModel(
        model=model,
        model_revision=os.getenv("IP_ASR_MODEL_REVISION", "v2.0.4"),
        device="cpu",
        disable_update=True,
        disable_pbar=True,
        disable_log=True,
        cache_dir=str(cache_dir),
    )
    elapsed = time.monotonic() - started
    size = dir_size(cache_dir)
    print(f"[fetch] ready in {elapsed:.1f}s, {size / 1e6:.1f} MB under {cache_dir}", flush=True)
    for sub in sorted(p for p in cache_dir.iterdir() if p.is_dir()):
        print(f"[fetch]   {sub.name}: {dir_size(sub) / 1e6:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
