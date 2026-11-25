"""Value objects shared by the adapter, the pipeline and the local service."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """One audio slice of a longer recording, positioned on the global timeline."""

    chunk_id: str
    path: str
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


@dataclass(frozen=True)
class Word:
    start_ms: int
    end_ms: int
    text: str
    confidence: float | None = None


@dataclass(frozen=True)
class Segment:
    """A recognised span. Times are always milliseconds on the *global* timeline."""

    start_ms: int
    end_ms: int
    text: str
    confidence: float | None = None
    words: list[Word] = field(default_factory=list)
    chunk_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
        }
        if self.confidence is not None:
            out["confidence"] = round(float(self.confidence), 4)
        if self.words:
            out["words"] = [
                {"start_ms": w.start_ms, "end_ms": w.end_ms, "text": w.text,
                 **({"confidence": round(float(w.confidence), 4)} if w.confidence is not None else {})}
                for w in self.words
            ]
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any], offset_ms: int = 0,
                  chunk_id: str | None = None) -> "Segment":
        words = [
            Word(
                start_ms=int(w["start_ms"]) + offset_ms,
                end_ms=int(w["end_ms"]) + offset_ms,
                text=str(w.get("text", "")),
                confidence=_opt_float(w.get("confidence")),
            )
            for w in (raw.get("words") or [])
        ]
        return cls(
            start_ms=int(raw["start_ms"]) + offset_ms,
            end_ms=int(raw["end_ms"]) + offset_ms,
            text=str(raw.get("text", "")).strip(),
            confidence=_opt_float(raw.get("confidence")),
            words=words,
            chunk_id=chunk_id,
        )


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
