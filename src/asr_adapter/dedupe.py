"""Overlap de-duplication.

Chunks are cut with an overlap so no word is lost at a boundary; the price is that the
overlap region is transcribed twice. Two chunks can also disagree slightly on the same
audio, so matching is by *time overlap plus text similarity*, not by text equality.
"""
from __future__ import annotations

import re
from dataclasses import replace
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


def _normalised_index(text: str) -> list[int]:
    """Position in `text` of every non-punctuation character, in order."""
    return [i for i, ch in enumerate(text) if not PUNCT.match(ch)]


def shared_run(a: str, b: str, min_chars: int = 5) -> tuple[int, int, int]:
    """Longest common run of characters between two segments, in normalised coordinates.

    Returns (start_in_a, start_in_b, length), or (0, 0, 0) when nothing reaches `min_chars`.
    A boundary repeat is not always at the head of the later segment — a VAD split can put the
    repeated words in the middle — so the match is searched anywhere, not just as a suffix /
    prefix pair.
    """
    na, nb = normalise(a), normalise(b)
    matcher = SequenceMatcher(None, na, nb, autojunk=False)
    match = matcher.find_longest_match(0, len(na), 0, len(nb))
    if match.size < min_chars:
        return 0, 0, 0
    return match.a, match.b, match.size


def _cut(text: str, start: int, length: int) -> str:
    """Remove `length` normalised characters starting at normalised offset `start`."""
    idx = _normalised_index(text)
    if length <= 0 or start >= len(idx):
        return text
    raw_from = idx[start]
    raw_to = idx[start + length - 1] + 1
    return (text[:raw_from] + text[raw_to:]).strip(" ，。、；：！？,.;:!?")


def stitch_boundary_repeats(
    segments: list[Segment],
    *,
    slack_ms: int = 1000,
    min_chars: int = 5,
) -> tuple[list[Segment], int]:
    """Remove words that survived twice because two overlapping chunks both transcribed them.

    `dedupe_overlaps` drops a *whole* duplicated sentence, but when one copy sits inside a
    longer sentence the similarity test no longer fires and a few characters survive twice
    ("…开机灯一直闪" + "机灯一直闪，屏幕停在…"). This second pass finds the longest shared run
    between time-overlapping neighbours and cuts it from the later segment, so the boundary
    reads once. A segment left with nothing is dropped rather than kept as empty text.
    Returns the rewritten segments and how many were cut.
    """
    out: list[Segment] = []
    trimmed = 0
    for seg in segments:
        if out:
            prev = out[-1]
            if overlaps(prev, seg, slack_ms):
                _, at_b, size = shared_run(prev.text, seg.text, min_chars)
                if size:
                    rest = _cut(seg.text, at_b, size)
                    if normalise(rest):
                        seg = replace(seg, text=rest)
                        trimmed += 1
                    else:
                        trimmed += 1
                        continue  # nothing new in this segment; keep the timeline, drop the copy
        out.append(seg)
    return out, trimmed
