"""Tests run against an in-process stub of the local service: no model, no network."""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import httpx
import pytest

from asr_adapter.client import AsrClient, AsrError
from asr_adapter.config import ConfigError, Settings
from asr_adapter.dedupe import dedupe_overlaps, similarity
from asr_adapter.models import Chunk, Segment
from asr_adapter.pipeline import load_chunks, run_transcription

FIXTURES = Path(__file__).parent / "fixtures"


def make_wav(path: Path) -> Path:
    """Tiny valid RIFF/WAVE header + silence; the stub never decodes it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"\x00\x00" * 1600
    header = (
        b"RIFF" + (36 + len(data)).to_bytes(4, "little") + b"WAVEfmt "
        + (16).to_bytes(4, "little") + (1).to_bytes(2, "little") + (1).to_bytes(2, "little")
        + (16000).to_bytes(4, "little") + (32000).to_bytes(4, "little")
        + (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
        + b"data" + len(data).to_bytes(4, "little")
    )
    path.write_bytes(header + data)
    return path


class StubService:
    """Records every request and replays canned segments per chunk (matched by filename)."""

    def __init__(self, responses: dict[str, list[dict]] | None = None, fail_times: int = 0):
        self.responses = responses or {}
        self.fail_times = fail_times
        self.urls: list[str] = []
        self.inflight = 0
        self.max_inflight = 0
        self._lock = threading.Lock()
        self._failures_left = fail_times

    def handler(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.urls.append(str(request.url))
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            time.sleep(0.02)  # make overlap detectable in the concurrency test
            if request.url.path == "/healthz":
                return httpx.Response(200, json={"status": "ok"})
            if self._failures_left > 0:
                with self._lock:
                    self._failures_left -= 1
                return httpx.Response(500, json={"detail": "boom"})
            name = None
            body = request.read()
            for candidate in self.responses:
                if candidate.encode() in body:
                    name = candidate
                    break
            segments = self.responses.get(name or "", [])
            return httpx.Response(200, json={"segments": segments})
        finally:
            with self._lock:
                self.inflight -= 1

    def client(self, settings: Settings) -> AsrClient:
        return AsrClient(settings, transport=httpx.MockTransport(self.handler), backoff=0.0)


def settings(**over) -> Settings:
    env = {"IP_ASR_URL": "http://127.0.0.1:8809", "IP_ASR_CONCURRENCY": "2"}
    env.update({k: str(v) for k, v in over.items()})
    return Settings.from_env(env)


# --- config -------------------------------------------------------------------

def test_missing_url_is_a_loud_error():
    with pytest.raises(ConfigError) as exc:
        Settings.from_env({})
    assert "IP_ASR_URL" in str(exc.value)


def test_bad_concurrency_rejected():
    with pytest.raises(ConfigError):
        Settings.from_env({"IP_ASR_URL": "http://127.0.0.1:8809", "IP_ASR_CONCURRENCY": "0"})


# --- offsets ------------------------------------------------------------------

def test_chunk_offset_is_added(tmp_path):
    wav = make_wav(tmp_path / "c001.wav")
    stub = StubService({"c001.wav": [{"start_ms": 1000, "end_ms": 2400, "text": "你好"}]})
    chunk = Chunk(chunk_id="c001", path=str(wav), start_ms=30_000, end_ms=60_000)
    segments = stub.client(settings()).transcribe_chunk(chunk)
    assert [(s.start_ms, s.end_ms) for s in segments] == [(31_000, 32_400)]
    assert segments[0].text == "你好"


# --- overlap de-duplication ---------------------------------------------------

def test_overlap_duplicate_removed_and_counted():
    a = Segment(9_000, 11_200, "我们下周把合同发过去", chunk_id="c001")
    b = Segment(10_400, 12_600, "我们下周把合同发过去。", chunk_id="c002")
    c = Segment(12_600, 14_000, "好的我这边确认一下", chunk_id="c002")
    kept, removed = dedupe_overlaps([a, c, b])
    assert removed == 1
    assert [s.text for s in kept] == ["我们下周把合同发过去", "好的我这边确认一下"]


def test_time_overlap_but_different_text_is_kept():
    a = Segment(9_000, 11_000, "报价按季度结算", chunk_id="c001")
    b = Segment(10_500, 12_000, "发票开给总部", chunk_id="c002")
    kept, removed = dedupe_overlaps([a, b])
    assert removed == 0 and len(kept) == 2


def test_normalisation_ignores_punctuation():
    assert similarity("我们下周把合同发过去。", "我们下周把合同发过去") == 1.0


# --- retries ------------------------------------------------------------------

def test_retry_then_success(tmp_path):
    wav = make_wav(tmp_path / "c001.wav")
    stub = StubService({"c001.wav": [{"start_ms": 0, "end_ms": 900, "text": "测试"}]},
                       fail_times=2)
    client = stub.client(settings(IP_ASR_RETRIES="2"))
    chunk = Chunk("c001", str(wav), 0, 10_000)
    segments = client.transcribe_chunk(chunk)
    assert len(segments) == 1
    assert client.request_count == 3


def test_retries_exhausted_reports_chunk(tmp_path):
    wav = make_wav(tmp_path / "c001.wav")
    stub = StubService({"c001.wav": []}, fail_times=10)
    client = stub.client(settings(IP_ASR_RETRIES="1"))
    with pytest.raises(AsrError) as exc:
        client.transcribe_chunk(Chunk("c001", str(wav), 0, 10_000))
    assert exc.value.chunk_id == "c001"
    assert client.request_count == 2


def test_client_error_is_not_retried(tmp_path):
    wav = make_wav(tmp_path / "c001.wav")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "bad audio"})

    client = AsrClient(settings(IP_ASR_RETRIES="3"),
                       transport=httpx.MockTransport(handler), backoff=0.0)
    with pytest.raises(AsrError):
        client.transcribe_chunk(Chunk("c001", str(wav), 0, 10_000))
    assert client.request_count == 1


# --- pipeline: concurrency, egress, partial failure ---------------------------

def test_concurrency_cap_respected(tmp_path):
    chunks = []
    responses = {}
    for i in range(6):
        wav = make_wav(tmp_path / f"c{i:03d}.wav")
        name = f"c{i:03d}.wav"
        chunks.append(Chunk(f"c{i:03d}", str(wav), i * 20_000, (i + 1) * 20_000))
        responses[name] = [{"start_ms": 0, "end_ms": 1500, "text": f"第{i}段"}]
    stub = StubService(responses)
    st = settings(IP_ASR_CONCURRENCY="2")
    transcript = run_transcription(chunks, settings=st, client=stub.client(st))
    assert stub.max_inflight <= 2
    assert transcript.chunks_ok == 6 and not transcript.errors
    assert [s.text for s in transcript.segments] == [f"第{i}段" for i in range(6)]


def test_only_local_service_is_contacted(tmp_path):
    wav = make_wav(tmp_path / "c001.wav")
    stub = StubService({"c001.wav": [{"start_ms": 0, "end_ms": 800, "text": "本地"}]})
    st = settings()
    run_transcription([Chunk("c001", str(wav), 0, 10_000)], settings=st, client=stub.client(st))
    assert stub.urls, "stub was never called"
    assert all(u.startswith("http://127.0.0.1:8809/") for u in stub.urls)


def test_partial_failure_keeps_good_chunks(tmp_path):
    good = make_wav(tmp_path / "good.wav")
    missing = tmp_path / "missing.wav"          # never created: client raises AsrError
    chunks = [Chunk("good", str(good), 0, 20_000), Chunk("missing", str(missing), 20_000, 40_000)]
    stub = StubService({"good.wav": [{"start_ms": 0, "end_ms": 1200, "text": "保留"}]})
    st = settings()
    transcript = run_transcription(chunks, settings=st, client=stub.client(st))
    assert transcript.chunks_ok == 1
    assert [e["chunk_id"] for e in transcript.errors] == ["missing"]
    assert transcript.text == "保留"


# --- privacy: nothing spoken may reach the logs -------------------------------

def test_transcript_text_never_logged(tmp_path, caplog):
    wav = make_wav(tmp_path / "c001.wav")
    secret = "这段话不能被写进日志"
    stub = StubService({"c001.wav": [{"start_ms": 0, "end_ms": 1200, "text": secret}]})
    caplog.set_level(logging.DEBUG)
    st = settings()
    transcript = run_transcription([Chunk("c001", str(wav), 0, 10_000)],
                                   settings=st, client=stub.client(st))
    assert transcript.text == secret
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert secret not in joined
    assert "c001" in joined          # identifiers are fine, content is not


def test_load_chunks_resolves_relative_paths(tmp_path):
    (tmp_path / "audio").mkdir()
    wav = make_wav(tmp_path / "audio" / "c001.wav")
    doc = {"chunks": [{"chunk_id": "c001", "path": "audio/c001.wav",
                       "start_ms": 0, "end_ms": 30_000}]}
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    chunks = load_chunks(path)
    assert chunks[0].path == str(wav.resolve())
