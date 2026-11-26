#!/usr/bin/env bash
# End-to-end run against the *real* local model, over every committed fixture:
# start the service once, then split / transcribe / score each recording separately and
# keep the raw artefacts under EVIDENCE/. The run fails (non-zero exit) if any fixture
# misses its CER ceiling or shows an undeduplicated overlap.
#
# Preconditions（前置条件）: torch + torchaudio + funasr installed in .venv, model weights present
# under $IP_ASR_MODEL_DIR (see scripts/fetch_model.py). Nothing here reaches the network.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export IP_ASR_MODEL_DIR="${IP_ASR_MODEL_DIR:-$PWD/models}"
# A local directory keeps the runtime offline; falls back to the ModelScope name if absent.
if [ -z "${IP_ASR_MODEL:-}" ] && [ -d "$IP_ASR_MODEL_DIR/paraformer-zh" ]; then
  export IP_ASR_MODEL="$IP_ASR_MODEL_DIR/paraformer-zh"
fi
if [ -z "${IP_ASR_VAD_MODEL:-}" ] && [ -d "$IP_ASR_MODEL_DIR/fsmn-vad" ]; then
  export IP_ASR_VAD_MODEL="$IP_ASR_MODEL_DIR/fsmn-vad"
fi
if [ -z "${IP_ASR_PUNC_MODEL:-}" ] && [ -d "$IP_ASR_MODEL_DIR/ct-punc-c" ]; then
  export IP_ASR_PUNC_MODEL="$IP_ASR_MODEL_DIR/ct-punc-c"
fi
PORT="${PORT:-8809}"
MAX_CER="${MAX_CER:-0.08}"
FIXDIR="eval/fixtures"
STAMP="$(date +%Y%m%d-%H%M)"
EVID="EVIDENCE/run-${STAMP}"
mkdir -p "$EVID" out

echo "== model dir: $IP_ASR_MODEL_DIR ($(du -sh "$IP_ASR_MODEL_DIR" 2>/dev/null | cut -f1))"
echo "== fixtures: $(.venv/bin/python -c "import json;d=json.load(open('$FIXDIR/index.json'));print(len(d['fixtures']))") recording(s), CER ceiling ${MAX_CER}"

echo "== starting service on :$PORT"
.venv/bin/python -m uvicorn service.server:app --host 127.0.0.1 --port "$PORT" \
  --log-level info > "$EVID/service.log" 2>&1 &
SERVICE=$!
trap 'kill $SERVICE 2>/dev/null || true' EXIT

for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/healthz" > "$EVID/healthz.json" 2>/dev/null; then
    echo "== healthy after ${i}s: $(cat "$EVID/healthz.json")"
    break
  fi
  sleep 1
  if [ "$i" = 180 ]; then echo "service did not become healthy" >&2; exit 1; fi
done

FAILED=0
ROWS=""
while read -r FID AUDIO TRUTH; do
  [ -z "${FID:-}" ] && continue
  DUR=$(.venv/bin/python -c "import json;print(round(json.load(open('$TRUTH'))['duration_ms']/1000,1))")
  HW=$(.venv/bin/python -c "import json;print(','.join(json.load(open('$TRUTH'))['hotwords']))")
  echo
  echo "== [$FID] $AUDIO  ${DUR}s  -> data/$FID + out/$FID"
  .venv/bin/python scripts/split_audio.py "$FIXDIR/$AUDIO" "data/$FID" --chunk 30 --overlap 2
  mkdir -p "$EVID/$FID"

  IP_ASR_URL="http://127.0.0.1:${PORT}" IP_ASR_CONCURRENCY=2 IP_ASR_RETRIES=2 \
    .venv/bin/python -m asr_adapter.cli \
    --chunks "data/$FID/chunks.json" --audio-root "data/$FID" --out "out/$FID/transcript.json" \
    --hotwords "$HW" > "$EVID/$FID/adapter.log" 2>&1 &
  CLI=$!
  PEERS=()
  while kill -0 "$CLI" 2>/dev/null; do
    while read -r peer; do PEERS+=("$peer"); done < <(ss -tnp 2>/dev/null | awk -v p="$CLI" '$0 ~ ("pid=" p ",") {print $5}')
    sleep 0.3
  done
  wait "$CLI" || echo "adapter exit=$? (see $EVID/$FID/adapter.log)"
  UNIQUE=$(printf '%s\n' "${PEERS[@]:-}" | grep -v '^$' | sort -u || true)
  printf '%s\n' "${UNIQUE:-none captured}" > "$EVID/$FID/peers_adapter.txt"

  cp "out/$FID/transcript.json" "$EVID/$FID/transcript.json"
  set +e
  .venv/bin/python eval/compute_cer.py --transcript "out/$FID/transcript.json" \
    --truth "$TRUTH" --max-cer "$MAX_CER" --out "$EVID/$FID/cer_report.md" \
    | tee "$EVID/$FID/cer_stdout.txt"
  RC=${PIPESTATUS[0]}
  set -e
  CER=$(.venv/bin/python -c "import json;print(f\"{json.load(open('$EVID/$FID/cer_report.json'))['cer_overall']*100:.2f}\")")
  if [ "$RC" = 0 ]; then VERDICT=PASS; else VERDICT=FAIL; FAILED=$((FAILED+1)); fi
  ROWS="${ROWS}${FID}|${AUDIO}|${DUR}|${CER}|${VERDICT}"$'\n'
  echo "== [$FID] CER ${CER}%  ->  ${VERDICT} (exit ${RC})"
done < <(.venv/bin/python -c "
import json
d = json.load(open('$FIXDIR/index.json'))
for f in d['fixtures']:
    print(f['id'], f['audio'], '$FIXDIR/' + f['truth'])
")

echo
echo "==================== summary ===================="
printf '%s' "$ROWS" | .venv/bin/python -c "
import json, sys
rows = [r.split('|') for r in sys.stdin.read().strip().splitlines() if r.strip()]
out = {'max_cer': float('$MAX_CER'), 'runs': [
    {'id': r[0], 'audio': r[1], 'duration_s': float(r[2]), 'cer': float(r[3]), 'verdict': r[4]} for r in rows]}
json.dump(out, open('$EVID/summary.json', 'w'), ensure_ascii=False, indent=2)
print(f\"{'fixture':8} {'audio':16} {'len':>8} {'CER':>8}  verdict\")
for r in out['runs']:
    print(f\"{r['id']:8} {r['audio']:16} {r['duration_s']:>7.1f}s {r['cer']:>7.2f}%  {r['verdict']}\")
print(f\"{len(out['runs'])} recordings, {sum(1 for r in out['runs'] if r['verdict']=='PASS')} passed\")
"
echo "== service log tail"
tail -3 "$EVID/service.log"
echo "== artefacts in $EVID"
ls -la "$EVID"
echo
echo "== offline evidence is produced separately: bash scripts/run_offline_evidence.sh"

if [ "$FAILED" != 0 ]; then
  echo "FAILED: $FAILED fixture(s) below the bar" >&2
  exit 1
fi
echo "ALL FIXTURES PASS (CER <= ${MAX_CER})"
