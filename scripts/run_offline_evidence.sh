#!/usr/bin/env bash
# Offline evidence: build the ASR image, run it with `--network none`, transcribe a fixture,
# and keep the raw packet capture that proves nothing left the container.
#
#   bash scripts/run_offline_evidence.sh
#
# Artefacts: EVIDENCE/offline-<stamp>/{net_ifaces.txt,egress.pcap,egress_packets.txt,
#            egress_summary.json,transcript.json,service.log,adapter.log,run.log}
set -euo pipefail

cd "$(dirname "$0")/.."
STAMP="$(date +%Y%m%d-%H%M)"
EVID="EVIDENCE/offline-${STAMP}"
IMAGE="${IMAGE:-ip-asr-offline:latest}"
mkdir -p "$EVID"
chmod +x scripts/offline_entrypoint.sh

echo "== building $IMAGE (network needed only for pip + base image)"
docker build --network host -f Dockerfile.asr -t "$IMAGE" . 2>&1 | tee "$EVID/build.log" | tail -15

echo "== running with --network none (this is the offline condition)"
set +e
docker run --rm --network none -e OUT=/out -v "$PWD/$EVID:/out" "$IMAGE" \
  bash /app/scripts/offline_entrypoint.sh 2>&1 | tee "$EVID/run.log"
RC=${PIPESTATUS[0]}
set -e

echo "== results in $EVID"
ls -la "$EVID"
echo
echo "-- egress summary --"
cat "$EVID/egress_summary.json" 2>/dev/null || echo "missing egress_summary.json"
echo
echo "-- transcript size --"
.venv/bin/python - "$EVID/transcript.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit("no transcript produced inside the offline container")
doc = json.loads(p.read_text(encoding="utf-8"))
print(f"segments={len(doc.get('segments') or [])} chunks={len(doc.get('chunk_timings') or {})} "
      f"failed_chunks={(doc.get('stats') or {}).get('chunks_failed')} "
      f"chars={sum(len(s['text']) for s in doc.get('segments') or [])}")
PY

echo "== scoring the offline transcript against the same ground truth"
.venv/bin/python eval/compute_cer.py --transcript "$EVID/transcript.json" \
  --truth eval/fixtures/ground_truth_b.json --max-cer 0.08 --out "$EVID/cer_report.md" \
  | tee "$EVID/cer_stdout.txt"

if [ "$RC" != 0 ]; then
  echo "offline container run exited $RC" >&2
  exit "$RC"
fi
echo "OFFLINE EVIDENCE DONE -> $EVID"
