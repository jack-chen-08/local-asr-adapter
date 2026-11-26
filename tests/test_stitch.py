"""Boundary stitching: words that two overlapping chunks both transcribed must appear once."""
from __future__ import annotations

from asr_adapter.dedupe import (
    dedupe_overlaps,
    shared_run,
    stitch_boundary_repeats,
)
from asr_adapter.models import Segment


def test_shared_run_finds_the_longest_common_stretch():
    a_at, b_at, size = shared_run("现在开机灯一直闪", "机灯一直闪，屏幕停在启动画面")
    assert size == 5
    assert (a_at, b_at) == (3, 0)
    assert shared_run("完全不同的话", "接下来另一句") == (0, 0, 0)
    assert shared_run("短", "短") == (0, 0, 0)  # below min_chars


def test_stitch_trims_repeated_head_and_keeps_the_rest():
    segs = [
        Segment(start_ms=28_000, end_ms=31_000, text="现在开机灯一直闪"),
        Segment(start_ms=30_500, end_ms=34_000, text="机灯一直闪，屏幕停在启动画面"),
    ]
    out, trimmed = stitch_boundary_repeats(segs)
    assert trimmed == 1
    assert out[0].text == "现在开机灯一直闪"
    assert out[1].text == "屏幕停在启动画面"
    assert out[1].start_ms == 30_500  # timeline untouched


def test_stitch_cuts_a_repeat_in_the_middle_of_the_later_segment():
    segs = [
        Segment(start_ms=28_000, end_ms=34_000, text="周六下午多功能厅已经有一场亲子手工"),
        Segment(start_ms=30_000, end_ms=36_000, text="多功能厅已经有一场，下午三点半结束"),
    ]
    out, trimmed = stitch_boundary_repeats(segs)
    assert trimmed == 1
    assert out[1].text == "下午三点半结束"


def test_stitch_leaves_non_overlapping_segments_alone():
    segs = [
        Segment(start_ms=0, end_ms=2_000, text="第一句话在这里"),
        Segment(start_ms=9_000, end_ms=12_000, text="第一句话在别处重复也会保留"),
    ]
    out, trimmed = stitch_boundary_repeats(segs)
    assert trimmed == 0
    assert [s.text for s in out] == [s.text for s in segs]


def test_fully_repeated_segment_is_dropped_not_emptied():
    segs = [
        Segment(start_ms=1_000, end_ms=4_000, text="多功能厅已经有一场亲子手工"),
        Segment(start_ms=3_500, end_ms=5_000, text="多功能厅已经有一场"),
    ]
    out, trimmed = stitch_boundary_repeats(segs)
    assert trimmed == 1
    assert len(out) == 1
    assert out[0].text == "多功能厅已经有一场亲子手工"


def test_pipeline_pair_removes_whole_copy_then_stitches_partial_one():
    segs = [
        Segment(start_ms=0, end_ms=3_000, text="周六下午多功能厅已经有一场亲子手工"),
        Segment(start_ms=2_800, end_ms=5_000, text="多功能厅已经有一场亲子手工"),
        Segment(start_ms=2_700, end_ms=5_600, text="多功能厅已经有一场，下午三点半结束"),
    ]
    kept, removed = dedupe_overlaps(segs)
    kept, trimmed = stitch_boundary_repeats(kept)
    text = "".join(s.text for s in kept)
    assert removed == 1
    assert text.count("多功能厅已经有一场") == 1
