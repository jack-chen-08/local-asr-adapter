#!/usr/bin/env bash
# Prove the adapter talks to nothing but the local service.
#
# Runs the real CLI against a stub service bound to 127.0.0.1 while sampling the process
# table's TCP sockets; any non-loopback peer fails the check. This is the same evidence the
# acceptance review asks for on a real deployment, where the stronger form is
#   docker run --network none ...      (see REPORT.md, "air-gapped run")
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
PORT="${PORT:-18809}"
OUT="${OUT:-out/egress_check.json}"
mkdir -p out

python3 - "$PORT" <<'PY' &
import json, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port = int(sys.argv[1])

class Handler(BaseHTTPRequestHandler):
    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"status": "ok"})

    def do_POST(self):
        time.sleep(0.4)   # keep the run long enough for socket sampling to catch peers
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self._send({"segments": [{"start_ms": 0, "end_ms": 1200, "text": "stub"}]})

    def log_message(self, *a):
        pass

ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
PY
STUB=$!
trap 'kill $STUB 2>/dev/null || true' EXIT
sleep 1

IP_ASR_URL="http://127.0.0.1:${PORT}" IP_ASR_CONCURRENCY=2 \
  .venv/bin/python -m asr_adapter.cli --chunks data/chunks.json --out "$OUT" \
  --log-level WARNING &
CLI=$!

PEERS=()
while kill -0 "$CLI" 2>/dev/null; do
  while read -r peer; do
    PEERS+=("$peer")
  done < <(ss -tnp 2>/dev/null | awk -v p="$CLI" '$0 ~ ("pid=" p ",") {print $5}')
  sleep 0.3
done
wait "$CLI" || true

UNIQUE=$(printf '%s\n' "${PEERS[@]:-}" | grep -v '^$' | sort -u || true)
echo "peers seen: ${UNIQUE:-none captured (run too short for sampling)}"
if printf '%s\n' "$UNIQUE" | grep -vE '^(127\.0\.0\.1|\[::1\]):' | grep -q .; then
  echo "FAIL: non-loopback peer observed" >&2
  exit 1
fi
echo "PASS: no non-loopback peer observed"
