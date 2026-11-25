"""Score a draft transcript against the fixture's ground truth.

Metrics
  CER          character error rate over the whole dialogue (punctuation stripped)
  per-turn CER where each ground-truth turn is matched to the hypotheses that overlap it
  hotword recall  share of the ten domain terms that survive transcription
  RTF          seconds of compute per second of audio, both summed-per-chunk and wall clock
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import editdistance

KEEP = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")


def norm(text: str) -> str:
    return KEEP.sub("", text).lower()


def cer(ref: str, hyp: str) -> float:
    ref, hyp = norm(ref), norm(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    return editdistance.eval(ref, hyp) / len(ref)


def per_turn(turns: list[dict], segments: list[dict]) -> list[dict]:
    rows = []
    for turn in turns:
        hits = [
            s for s in segments
            if s["start_ms"] < turn["end_ms"] + 500 and s["end_ms"] > turn["start_ms"] - 500
        ]
        hyp = "".join(s["text"] for s in hits)
        rows.append(
            {
                "speaker": turn["speaker"],
                "start_ms": turn["start_ms"],
                "ref_chars": len(norm(turn["text"])),
                "hyp_chars": len(norm(hyp)),
                "cer": round(cer(turn["text"], hyp), 4),
                "ref": turn["text"],
                "hyp": hyp,
            }
        )
    return rows


def hotword_recall(hotwords: list[str], ref_text: str, hyp_text: str) -> list[dict]:
    rows = []
    for word in hotwords:
        ref_hits = norm(ref_text).count(norm(word))
        hyp_hits = norm(hyp_text).count(norm(word))
        rows.append(
            {
                "hotword": word,
                "in_reference": ref_hits,
                "in_transcript": hyp_hits,
                "recalled": hyp_hits >= ref_hits and ref_hits > 0,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--truth", type=Path, default=Path("eval/fixtures/ground_truth.json"))
    parser.add_argument("--out", type=Path, default=None, help="write a markdown report here")
    args = parser.parse_args()

    doc = json.loads(args.transcript.read_text(encoding="utf-8"))
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    segments = doc.get("segments") or []
    hyp_text = "".join(s["text"] for s in segments)
    ref_text = "".join(t["text"] for t in truth["segments"])

    overall = cer(ref_text, hyp_text)
    turns = per_turn(truth["segments"], segments)
    hw = hotword_recall(truth.get("hotwords") or [], ref_text, hyp_text)

    timings = sorted((doc.get("chunk_timings") or {}).values())
    audio_s = truth["duration_ms"] / 1000.0
    stats = doc.get("stats") or {}
    timing = {
        "chunks": len(timings),
        "p50_s": round(statistics.median(timings), 2) if timings else None,
        "p95_s": round(timings[int(len(timings) * 0.95) - 1], 2) if timings else None,
        "max_s": round(max(timings), 2) if timings else None,
        "sum_s": round(sum(timings), 2) if timings else None,
        "wall_s": stats.get("elapsed_s"),
        "audio_s": round(audio_s, 2),
        "rtf_summed": round(sum(timings) / audio_s, 3) if timings else None,
        "rtf_wall": round(stats["elapsed_s"] / audio_s, 3) if stats.get("elapsed_s") else None,
        "concurrency": stats.get("concurrency"),
    }

    result = {
        "cer_overall": round(overall, 4),
        "ref_chars": len(norm(ref_text)),
        "hyp_chars": len(norm(hyp_text)),
        "segments": len(segments),
        "duplicates_removed": stats.get("removed_duplicates"),
        "failed_chunks": stats.get("chunks_failed"),
        "hotword_recall": round(sum(1 for h in hw if h["recalled"]) / len(hw), 3) if hw else None,
        "hotwords": hw,
        "timing": timing,
        "turns": turns,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "turns"}, ensure_ascii=False, indent=2))

    if args.out:
        lines = [
            "# Transcription evaluation",
            "",
            f"- audio: {audio_s:.1f} s, {len(truth['segments'])} turns, {len(segments)} recognised segments",
            f"- **CER overall: {overall*100:.2f}%**  ({len(norm(ref_text))} reference characters)",
            f"- hotword recall: {result['hotword_recall']*100:.0f}% ({sum(1 for h in hw if h['recalled'])}/{len(hw)})",
            f"- duplicate segments removed by overlap logic: {stats.get('removed_duplicates')}",
            f"- failed chunks: {stats.get('chunks_failed')}",
            "",
            "## Timing",
            "",
            "| chunk p50 | chunk p95 | chunk max | sum of chunk time | wall clock | audio | RTF (summed) | RTF (wall) |",
            "|---|---|---|---|---|---|---|---|",
            f"| {timing['p50_s']} s | {timing['p95_s']} s | {timing['max_s']} s | {timing['sum_s']} s | "
            f"{timing['wall_s']} s | {timing['audio_s']} s | {timing['rtf_summed']} | {timing['rtf_wall']} |",
            "",
            "## Hotwords",
            "",
            "| term | in reference | in transcript | recalled |",
            "|---|---|---|---|",
        ]
        for row in hw:
            lines.append(
                f"| {row['hotword']} | {row['in_reference']} | {row['in_transcript']} | "
                f"{'yes' if row['recalled'] else '**no**'} |"
            )
        lines += ["", "## Per-turn CER", "", "| # | speaker | CER | reference | hypothesis |", "|---|---|---|---|---|"]
        for i, row in enumerate(turns, 1):
            lines.append(
                f"| {i} | {row['speaker']} | {row['cer']*100:.1f}% | {row['ref']} | {row['hyp']} |"
            )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nreport -> {args.out}")
        args.out.with_suffix(".json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
