#!/usr/bin/env bash
# Runs *inside* the ASR container with `--network none`.
#
# Evidence produced under $OUT (a mounted directory):
#   net_ifaces.txt      interfaces visible in the container (loopback only expected)
#   egress.pcap         raw capture taken while the transcription ran
#   egress_packets.txt  tcpdump -nr rendering of that capture
#   egress_summary.json total packets / non-loopback packets / verdict
#   transcript.json     the transcript produced with no network available
#   service.log, healthz.json
set -uo pipefail
OUT="${OUT:-/out}"
PORT="${PORT:-8809}"
mkdir -p "$OUT" /tmp/chunks

echo "== interfaces in the container"
{
  echo "# interfaces (docker run --network none leaves only loopback)"
  ip -o addr 2>/dev/null || cat /proc/net/dev
  echo
  echo "# reachability probe (expected to fail: no route out)"
  curl -m 3 -sS https://example.com -o /dev/null; echo "curl_rc=$?"
} | tee "$OUT/net_ifaces.txt"

echo "== starting tcpdump"
tcpdump -i any -s 0 -w "$OUT/egress.pcap" 2> "$OUT/tcpdump.log" &
TP=$!
sleep 2

echo "== starting ASR service"
uvicorn server:app --host 127.0.0.1 --port "$PORT" --workers 1 > "$OUT/service.log" 2>&1 &
SVC=$!
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/healthz" > "$OUT/healthz.json" 2>/dev/null; then
    echo "== healthy after ${i}s: $(cat "$OUT/healthz.json")"
    break
  fi
  sleep 1
  [ "$i" = 180 ] && { echo "service did not become healthy" >&2; }
done

HW=$(python -c "import json;print(','.join(json.load(open('fixtures/ground_truth_b.json'))['hotwords']))")
python scripts/split_audio.py fixtures/dialogue_b.wav /tmp/chunks --chunk 30 --overlap 2
echo "== transcribing with no network"
IP_ASR_URL="http://127.0.0.1:${PORT}" IP_ASR_CONCURRENCY=2 IP_ASR_RETRIES=2 \
  python -m asr_adapter.cli --chunks /tmp/chunks/chunks.json --out "$OUT/transcript.json" \
  --hotwords "$HW" > "$OUT/adapter.log" 2>&1
echo "adapter_rc=$?"
if [ ! -s "$OUT/transcript.json" ]; then
  echo "FATAL: no transcript written, see $OUT/adapter.log" >&2
  tail -20 "$OUT/adapter.log" >&2
fi

kill -INT "$TP" 2>/dev/null
wait "$TP" 2>/dev/null
kill "$SVC" 2>/dev/null

echo "== packet record"
tcpdump -nr "$OUT/egress.pcap" 2>/dev/null > "$OUT/egress_packets.txt" || true
python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
lines = [ln for ln in (out / "egress_packets.txt").read_text(errors="replace").splitlines() if ln.strip()]
loop = ("127.0.0.1", "::1", "IP6 ::1")
non_loop = [ln for ln in lines if not any(m in ln for m in loop)]
summary = {
    "packets_total": len(lines),
    "packets_non_loopback": len(non_loop),
    "non_loopback_samples": non_loop[:5],
    "verdict": "PASS" if non_loop == [] and lines else "FAIL",
    "note": "capture taken inside a container started with --network none",
}
(out / "egress_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
