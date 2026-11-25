"""HTTP client for the in-cloud inference service.

The contract with the local service is deliberately small:

    POST {IP_ASR_URL}/transcribe      multipart/form-data
        file         audio bytes (wav)
        hotwords     JSON array of strings (may be empty)
        sample_rate  integer
    ->  {"segments": [{"start_ms", "end_ms", "text", "confidence"?, "words"?}, ...]}

The service returns times relative to the *chunk*; this client adds the chunk offset so
callers only ever see global millisecond timestamps.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import httpx

from .config import Settings
from .models import Chunk, Segment


class AsrError(RuntimeError):
    """The inference service could not produce segments for this chunk."""

    def __init__(self, reason: str, *, chunk_id: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.chunk_id = chunk_id


class AsrClient:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None,
                 backoff: float = 0.25) -> None:
        self._settings = settings
        self._backoff = backoff
        self._client = httpx.Client(transport=transport, timeout=settings.timeout_s)
        self.request_count = 0

    # -- lifecycle ---------------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AsrClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def healthy(self) -> bool:
        try:
            response = self._client.get(f"{self._settings.base_url}/healthz")
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    # -- the one call the ticket asks for -----------------------------------------
    def transcribe_chunk(self, chunk: Chunk, hotwords: Iterable[str] | None = None) -> list[Segment]:
        """Transcribe one chunk and shift its timestamps onto the global timeline."""
        path = Path(chunk.path)
        if not path.is_file():
            raise AsrError("chunk file not found", chunk_id=chunk.chunk_id)
        audio = path.read_bytes()
        if not audio:
            raise AsrError("chunk file is empty", chunk_id=chunk.chunk_id)
        payload = self._post_with_retry(
            chunk_id=chunk.chunk_id,
            files={"file": (path.name, audio, "audio/wav")},
            data={
                "hotwords": json.dumps(list(hotwords or []), ensure_ascii=False),
                "sample_rate": str(self._settings.sample_rate),
            },
        )
        raw_segments = payload.get("segments")
        if not isinstance(raw_segments, list):
            raise AsrError("service response has no segments list", chunk_id=chunk.chunk_id)
        segments = [
            Segment.from_dict(item, offset_ms=chunk.start_ms, chunk_id=chunk.chunk_id)
            for item in raw_segments
            if isinstance(item, dict) and str(item.get("text", "")).strip()
        ]
        segments.sort(key=lambda s: (s.start_ms, s.end_ms))
        return segments

    # -- transport ---------------------------------------------------------------
    def _post_with_retry(self, *, chunk_id: str, files: dict, data: dict) -> dict:
        attempts = self._settings.retries + 1
        last: str = "unknown error"
        for attempt in range(attempts):
            self.request_count += 1
            try:
                response = self._client.post(
                    f"{self._settings.base_url}/transcribe", files=files, data=data
                )
            except httpx.HTTPError as exc:            # network: retry
                last = f"transport error: {type(exc).__name__}"
            else:
                if response.status_code == 200:
                    try:
                        payload = response.json()
                    except ValueError:
                        last = "service returned non-JSON body"
                    else:
                        if isinstance(payload, dict):
                            return payload
                        last = "service returned unexpected JSON"
                elif response.status_code == 429 or response.status_code >= 500:
                    last = f"service http {response.status_code}"   # retry
                else:
                    # 4xx from our own service is a contract problem: retrying cannot help
                    raise AsrError(
                        f"service rejected chunk: http {response.status_code}",
                        chunk_id=chunk_id,
                    )
            if attempt < attempts - 1:
                time.sleep(self._backoff * (2 ** attempt))
        raise AsrError(f"transcription failed after {attempts} attempts: {last}",
                       chunk_id=chunk_id)
