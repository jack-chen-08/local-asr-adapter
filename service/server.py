"""Local inference service: FunASR Paraformer behind one endpoint.

This process runs *inside* the customer's cloud and never calls out. The model is baked
into the image at build time (see Dockerfile.asr) so the container can transcribe with no
network at all, which is what the acceptance criterion about an air-gapped run checks.

Audio bytes and recognised text are never logged: only counts, durations and identifiers.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import tempfile
import time
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

log = logging.getLogger("asr_service")
logging.basicConfig(level=os.getenv("ASR_LOG_LEVEL", "INFO"))

MODEL_NAME = os.getenv("IP_ASR_MODEL", "paraformer-zh")
MODEL_DIR = os.getenv("IP_ASR_MODEL_DIR", "/models")
DEVICE = os.getenv("IP_ASR_DEVICE", "cpu")
# VAD is what turns a 30 s chunk into sentence-level segments with timestamps; without it the
# engine returns one blob per chunk and the overlap region can never be de-duplicated.
VAD_MODEL = os.getenv("IP_ASR_VAD_MODEL", "fsmn-vad")
PUNC_MODEL = os.getenv("IP_ASR_PUNC_MODEL", "ct-punc-c")
VAD_MAX_SEGMENT_MS = int(os.getenv("IP_ASR_VAD_MAX_SEGMENT_MS", "30000"))

app = FastAPI(title="local-asr-service", version="1.0.0")
_model: Any = None
_loaded_at: float | None = None


def _load_model() -> Any:
    global _model, _loaded_at
    if _model is not None:
        return _model
    from funasr import AutoModel  # imported lazily so /healthz answers before load

    started = time.monotonic()
    kwargs: dict[str, Any] = dict(
        model=MODEL_NAME,
        model_revision="v2.0.4",
        device=DEVICE,
        disable_update=True,
        disable_pbar=True,
        disable_log=True,
        cache_dir=MODEL_DIR,
    )
    if VAD_MODEL:
        kwargs["vad_model"] = VAD_MODEL
        kwargs["vad_kwargs"] = {"max_single_segment_time": VAD_MAX_SEGMENT_MS}
    if PUNC_MODEL:
        kwargs["punc_model"] = PUNC_MODEL
    _model = AutoModel(**kwargs)
    _loaded_at = time.monotonic() - started
    log.info(
        "model %s + vad %s + punc %s loaded in %.1fs on %s",
        MODEL_NAME, VAD_MODEL or "-", PUNC_MODEL or "-", _loaded_at, DEVICE,
    )
    return _model


@app.get("/healthz")
def healthz() -> JSONResponse:
    """Cheap liveness probe: does not load the model, so compose can start fast."""
    return JSONResponse(
        {
            "status": "ok",
            "model": MODEL_NAME,
            "device": DEVICE,
            "model_loaded": _model is not None,
            "load_seconds": round(_loaded_at, 2) if _loaded_at else None,
        }
    )


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    hotwords: str = Form("[]"),
    sample_rate: int = Form(16000),
) -> JSONResponse:
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="empty audio")
    try:
        words = json.loads(hotwords) if hotwords else []
        if not isinstance(words, list):
            raise ValueError
    except ValueError:
        raise HTTPException(status_code=400, detail="hotwords must be a JSON array") from None

    model = _load_model()
    started = time.monotonic()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio)
        tmp.flush()
        result = model.generate(
            input=tmp.name,
            hotword=" ".join(str(w) for w in words) or None,
            batch_size_s=60,
            sentence_timestamp=True,
        )
    segments = _segments_from(result)
    log.info(
        "transcribed %d bytes, %d segments in %.2fs",
        len(audio), len(segments), time.monotonic() - started,
    )
    return JSONResponse({"segments": segments, "model": MODEL_NAME, "sample_rate": sample_rate})


CJK = "\u4e00-\u9fff"
_CJK_GAP = re.compile(rf"(?<=[{CJK}])\s+(?=[{CJK}])")


def _join_cjk(text: str) -> str:
    """Paraformer emits one token per space-separated piece; Chinese must be joined back.

    Spaces between two CJK characters are an artefact of the tokenizer, not the speaker.
    Spaces next to latin/digits are kept so "RTF 0.07" and "US$190" stay readable.
    """
    out = text.strip()
    prev = None
    while prev != out:  # a single pass already handles "a b c"; loop is belt-and-braces
        prev = out
        out = _CJK_GAP.sub("", out)
    return out


def _segments_from(result: Any) -> list[dict[str, Any]]:
    if not result:
        return []
    first = result[0] if isinstance(result, list) else result
    sentences = first.get("sentence_info") or []
    if sentences:
        return [
            {
                "start_ms": int(s.get("start", 0)),
                "end_ms": int(s.get("end", 0)),
                "text": _join_cjk(str(s.get("text", ""))),
                "confidence": _confidence(s),
            }
            for s in sentences
            if str(s.get("text", "")).strip()
        ]
    text = _join_cjk(str(first.get("text", "")))
    if not text:
        return []
    # no sentence timestamps from the engine: one segment spanning the chunk
    return [{"start_ms": 0, "end_ms": int(first.get("duration_ms") or 0), "text": text}]


def _confidence(sentence: dict[str, Any]) -> float | None:
    """Paraformer does not ship per-sentence confidence; expose the mean token score
    when the engine provides one, otherwise omit the field rather than invent a number."""
    tokens = sentence.get("tokens") or []
    scores = [t.get("score") for t in tokens if isinstance(t, dict) and t.get("score") is not None]
    if not scores:
        return None
    return round(sum(float(x) for x in scores) / len(scores), 4)
