"""Cut a long WAV into overlapping chunks and emit chunks.json.

Overlap exists so a word straddling a boundary is complete in at least one chunk; the
adapter de-duplicates the repeated region afterwards. Chunk times are global milliseconds,
which is the only timeline the adapter ever reports.

    python scripts/split_audio.py eval/fixtures/dialogue.wav data --chunk 30 --overlap 2
"""
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path


def split(src: Path, out_dir: Path, chunk_s: float, overlap_s: float) -> dict:
    with wave.open(str(src), "rb") as reader:
        params = reader.getparams()
        if params.sampwidth != 2 or params.nchannels != 1:
            raise SystemExit(f"{src}: expected 16-bit mono, got {params.nchannels}ch/{params.sampwidth*8}bit")
        sr = params.framerate
        total = reader.getnframes()
        frames = reader.readframes(total)

    chunk_frames = int(chunk_s * sr)
    step = int((chunk_s - overlap_s) * sr)
    if step <= 0:
        raise SystemExit("overlap must be smaller than chunk length")

    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    index = 0
    offset = 0
    while offset < total:
        end = min(offset + chunk_frames, total)
        audio = frames[offset * 2 : end * 2]
        name = f"chunk_{index:04d}.wav"
        with wave.open(str(out_dir / name), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(sr)
            writer.writeframes(audio)
        chunks.append(
            {
                "chunk_id": f"chunk_{index:04d}",
                "path": name,
                "start_ms": int(offset / sr * 1000),
                "end_ms": int(end / sr * 1000),
            }
        )
        index += 1
        if end >= total:
            break
        offset += step

    doc = {
        "source": src.name,
        "sample_rate": sr,
        "chunk_seconds": chunk_s,
        "overlap_seconds": overlap_s,
        "chunks": chunks,
    }
    (out_dir / "chunks.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return doc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--chunk", type=float, default=30.0)
    parser.add_argument("--overlap", type=float, default=2.0)
    args = parser.parse_args()
    doc = split(args.source, args.out_dir, args.chunk, args.overlap)
    print(f"{len(doc['chunks'])} chunks -> {args.out_dir/'chunks.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
