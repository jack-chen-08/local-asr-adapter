"""Overlap de-duplication.

Chunks are cut with an overlap so no word is lost at a boundary; the price is that the
overlap region is transcribed twice. Two chunks can also disagree slightly on the same
audio, so matching is by *time overlap plus text similarity*, not by text equality.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from .models import Segment

PUNCT = re.compile(r"[\s，。、；：！？,.;:!?\"'“”‘’（）()\-—…]+")


def normalise(text: str) -> str:
    return PUNCT.sub("", text)


def similarity(a: str, b: str) -> float:
    na, nb = normalise(a), normalise(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def overlaps(a: Segment, b: Segment, slack_ms: int = 0) -> bool:
    return a.start_ms - slack_ms < b.end_ms and b.start_ms - slack_ms < a.end_ms


def dedupe_overlaps(
    segments: list[Segment],
    *,
    similarity_threshold: float = 0.75,
    slack_ms: int = 1000,
    lookback: int = 12,
    min_overlap_ms: int = 200,
) -> tuple[list[Segment], int]:
    """Drop the later copy of a sentence that appears in two chunks.

    Returns the kept segments and how many were removed, so callers can report a number
    instead of silently shrinking the transcript.
    """
    kept: list[Segment] = []
    removed = 0
    for seg in sorted(segments, key=lambda s: (s.start_ms, s.end_ms)):
        duplicate = False
        for prev in reversed(kept[-lookback:]):
            if not overlaps(prev, seg, slack_ms):
                continue
            # A boundary cut can hand us half a sentence and its completion as two segments;
            # those are duplicates too, so containment counts as a match.
            shared_ms = min(prev.end_ms, seg.end_ms) - max(prev.start_ms, seg.start_ms)
            if shared_ms < min_overlap_ms:
                continue
            a, b = normalise(prev.text), normalise(seg.text)
            contained = bool(a) and bool(b) and (a in b or b in a)
            if contained or similarity(prev.text, seg.text) >= similarity_threshold:
                duplicate = True
                # keep the longer text: a boundary cut often truncates one side
                if len(normalise(seg.text)) > len(normalise(prev.text)):
                    kept[kept.index(prev)] = seg
                break
        if duplicate:
            removed += 1
        else:
            kept.append(seg)
    return kept, removed
