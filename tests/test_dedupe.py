"""De-duplication tests: time overlap is necessary, text similarity decides."""
from __future__ import annotations

from asr_adapter.dedupe import dedupe_overlaps
from asr_adapter.models import Segment


def seg(chunk_id: str, start_ms: int, end_ms: int, text: str) -> Segment:
    return Segment(chunk_id=chunk_id, start_ms=start_ms, end_ms=end_ms, text=text)


def test_same_chunk_overlapping_copy_is_dropped():
    """Two segments of *one* chunk can overlap when VAD splits a long breath group."""
    kept, removed = dedupe_overlaps([
        seg("c0", 56_130, 58_275, "属于异常到货可以拒收，"),
        seg("c0", 56_840, 57_895, "到货可以拒。"),
    ])
    assert removed == 1
    assert kept[0].text == "属于异常到货可以拒收，"


def test_containment_counts_as_duplicate_across_chunks():
    """A boundary cut hands over half a sentence plus its completion — keep the longer one."""
    kept, removed = dedupe_overlaps([
        seg("c0", 28_970, 30_605, "您方便报一下单号吗？"),
        seg("c1", 27_500, 29_400, "您方便报"),
    ])
    assert removed == 1
    assert kept == [seg("c0", 28_970, 30_605, "您方便报一下单号吗？")]


def test_unrelated_segments_in_the_overlap_window_survive():
    """Different words in the same time window are new content, not duplicates."""
    kept, removed = dedupe_overlaps([
        seg("c0", 0, 3_000, "请问有什么可以帮您？"),
        seg("c1", 1_000, 4_000, "单号是 KF 二零七六。"),
    ])
    assert removed == 0
    assert len(kept) == 2
