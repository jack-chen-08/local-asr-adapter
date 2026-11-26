"""One-off sanitiser: strip operator-side paths and the platform order id from
files that ship in the public repository.

The evidence tree is raw output of real runs; nothing about the measurements
changes here.  Only the host paths (``/home/<user>/clawhunt_delivery/<order>``)
and the platform order id in the report title / image tag are rewritten, so a
reader cloning the repository cannot learn our internal directory layout or
which platform ticket this was.

    python scripts/sanitize_evidence.py            # rewrite in place
    python scripts/sanitize_evidence.py --check    # report only, exit 1 if dirty

The rewrite is recorded in ``EVIDENCE/SANITIZED.md`` so the transformation is
reviewable and no one mistakes the scrubbed logs for untouched ones.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPLACEMENTS: List[Tuple[str, str]] = [
    ("/home/ubuntuuser/clawhunt_delivery/2284_asr_adapter", "<workdir>"),
    ("clawhunt_delivery/2284_asr_adapter", "<workdir>"),
    ("/home/ubuntuuser", "<home>"),
    ("asr-offline:2284", "asr-offline:latest"),
    ("（#2284）", ""),
    ("(#2284)", ""),
    ("#2284", ""),
]

TEXT_EXT = {".md", ".log", ".json", ".txt", ".py", ".sh", ".yml", ".yaml", ".toml", ".cfg", ".ini"}
SKIP_DIRS = {".git", ".venv", "models", "__pycache__", ".pytest_cache", "data", "out"}
SKIP_FILES = {"scripts/sanitize_evidence.py", "EVIDENCE/SANITIZED.md"}


def _iter_files() -> List[str]:
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), ROOT)
            if rel in SKIP_FILES:
                continue
            ext = os.path.splitext(name)[1]
            if ext in TEXT_EXT or name.startswith("Dockerfile"):
                out.append(os.path.join(dirpath, name))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    changed: List[str] = []
    hits: List[str] = []
    for path in _iter_files():
        try:
            with open(path, "r", encoding="utf-8", errors="strict") as fh:
                original = fh.read()
        except (UnicodeDecodeError, OSError):
            continue
        text = original
        for needle, repl in REPLACEMENTS:
            text = text.replace(needle, repl)
        if text == original:
            continue
        rel = os.path.relpath(path, ROOT)
        changed.append(rel)
        for needle, _ in REPLACEMENTS:
            if needle in original:
                hits.append("%s: %s" % (rel, needle))
        if not args.check:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)

    if args.check:
        if changed:
            print("dirty files (%d):" % len(changed))
            for c in changed:
                print("  " + c)
            return 1
        print("clean: no operator paths or order ids in shippable files")
        return 0

    print("rewritten %d file(s)" % len(changed))
    if changed:
        note = os.path.join(ROOT, "EVIDENCE", "SANITIZED.md")
        os.makedirs(os.path.dirname(note), exist_ok=True)
        with open(note, "w", encoding="utf-8") as fh:
            fh.write("# 证据脱敏记录\n\n"
                     "下列文件只做了路径与工单号替换，测量数据、时间戳、包数、CER 数值均未改动。\n\n"
                     "| 文件 | 替换内容 |\n|---|---|\n")
            for h in hits:
                fh.write("| `%s` | `%s` |\n" % tuple(h.split(": ", 1)))
        print("note: EVIDENCE/SANITIZED.md")
    for c in changed:
        print("  " + c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
