"""CLI: transcribe a chunk list into a draft transcript JSON.

Exit codes
  0  all chunks transcribed
  2  configuration problem (no IP_ASR_URL, bad file)
  3  partial: at least one chunk failed after retries (draft still written, for review)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .client import AsrClient
from .config import ConfigError, Settings
from .pipeline import load_chunks, run_transcription

log = logging.getLogger("asr_adapter.cli")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="asr_adapter")
    parser.add_argument("--chunks", required=True, help="chunks.json produced by the splitter")
    parser.add_argument("--out", required=True, help="where to write the draft transcript JSON")
    parser.add_argument("--audio-root", default=None, help="base dir for relative chunk paths")
    parser.add_argument("--hotwords", default=None, help="comma separated domain terms")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        settings = Settings.from_env()
        chunks = load_chunks(args.chunks, args.audio_root)
    except (ConfigError, ValueError, OSError, KeyError) as exc:
        log.error("cannot start: %s", exc)
        return 2

    hotwords = [h.strip() for h in (args.hotwords or "").split(",") if h.strip()]

    with AsrClient(settings) as client:
        if not client.healthy():
            log.error("inference service at %s is not healthy (/healthz failed)", settings.base_url)
            return 2
        transcript = run_transcription(
            chunks, settings=settings, client=client, hotwords=hotwords
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("draft written to %s", out)

    if transcript.errors:
        log.warning("%d of %d chunks failed; see the errors array", len(transcript.errors),
                    transcript.chunks_total)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
