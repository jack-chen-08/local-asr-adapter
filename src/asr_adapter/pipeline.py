"""Concurrent transcription of a chunk list, with global offsets and overlap de-duplication."""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .client import AsrClient, AsrError
from .config import Settings
from .dedupe import dedupe_overlaps, stitch_boundary_repeats
from .models import Chunk, Segment

log = logging.getLogger("asr_adapter.pipeline")


@dataclass
class Transcript:
    segments: list[Segment] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    chunks_total: int = 0
    chunks_ok: int = 0
    removed_duplicates: int = 0
    stitched_repeats: int = 0
    requests: int = 0
    elapsed_s: float = 0.0
    hotwords: list[str] = field(default_factory=list)
    concurrency: int = 0
    chunk_timings: dict[str, float] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.segments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft": True,
            "segments": [s.to_dict() for s in self.segments],
            "stats": {
                "chunks_total": self.chunks_total,
                "chunks_ok": self.chunks_ok,
                "chunks_failed": len(self.errors),
                "removed_duplicates": self.removed_duplicates,
                "stitched_repeats": self.stitched_repeats,
                "requests": self.requests,
                "elapsed_s": round(self.elapsed_s, 2),
                "concurrency": self.concurrency,
            },
            "errors": self.errors,
            "chunk_timings": {k: round(v, 3) for k, v in sorted(self.chunk_timings.items())},
        }


def load_chunks(path: str | Path, audio_root: str | Path | None = None) -> list[Chunk]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw.get("chunks") if isinstance(raw, dict) else raw
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: no chunks found")
    root = Path(audio_root) if audio_root else Path(path).parent
    out: list[Chunk] = []
    for item in items:
        audio = Path(item["path"])
        if not audio.is_absolute():
            audio = (root / audio).resolve()
        out.append(
            Chunk(
                chunk_id=str(item.get("chunk_id") or audio.stem),
                path=str(audio),
                start_ms=int(item["start_ms"]),
                end_ms=int(item["end_ms"]),
            )
        )
    out.sort(key=lambda c: c.start_ms)
    return out


def run_transcription(
    chunks: list[Chunk],
    *,
    settings: Settings,
    client: AsrClient | None = None,
    hotwords: Iterable[str] | None = None,
) -> Transcript:
    """Transcribe every chunk, keep global order, de-duplicate the overlap regions.

    A failed chunk never aborts the run: it is reported in ``errors`` so the draft can be
    reviewed and the chunk retried, which is what the review queue expects.
    """
    started = time.monotonic()
    hotword_list = [str(h) for h in (hotwords or [])]
    owned = client is None
    client = client or AsrClient(settings)
    transcript = Transcript(chunks_total=len(chunks), hotwords=hotword_list)
    collected: list[Segment] = []
    try:
        with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
            futures = {
                pool.submit(_timed_transcribe, client, chunk, hotword_list): chunk
                for chunk in chunks
            }
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    segments, seconds = future.result()
                except AsrError as exc:
                    transcript.errors.append(
                        {"chunk_id": chunk.chunk_id, "reason": exc.reason}
                    )
                    log.warning("chunk %s failed: %s", chunk.chunk_id, exc.reason)
                except Exception as exc:  # noqa: BLE001 - one bad chunk must not kill the run
                    transcript.errors.append(
                        {"chunk_id": chunk.chunk_id, "reason": type(exc).__name__}
                    )
                    log.warning("chunk %s crashed: %s", chunk.chunk_id, type(exc).__name__)
                else:
                    transcript.chunks_ok += 1
                    transcript.chunk_timings[chunk.chunk_id] = seconds
                    log.debug("chunk %s ok: %d segments", chunk.chunk_id, len(segments))
                    collected.extend(segments)
    finally:
        if owned:
            client.close()

    kept, removed = dedupe_overlaps(collected)
    kept, stitched = stitch_boundary_repeats(kept)
    transcript.segments = kept
    transcript.removed_duplicates = removed
    transcript.stitched_repeats = stitched
    transcript.requests = client.request_count
    transcript.elapsed_s = time.monotonic() - started
    transcript.concurrency = settings.concurrency
    log.info(
        "transcription done: chunks %d/%d ok, segments %d, duplicates removed %d, "
        "boundary repeats stitched %d, requests %d",
        transcript.chunks_ok, transcript.chunks_total, len(kept), removed, stitched,
        transcript.requests,
    )
    return transcript


def _timed_transcribe(
    client: AsrClient, chunk: Chunk, hotwords: list[str]
) -> tuple[list[Segment], float]:
    """Wall-clock seconds for one chunk, measured around the client call only."""
    started = time.monotonic()
    segments = client.transcribe_chunk(chunk, hotwords)
    return segments, time.monotonic() - started
