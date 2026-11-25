#!/usr/bin/env bash
# End-to-end run against the *real* local model: start the service, transcribe the fixture,
# score it, and keep the raw artefacts under EVIDENCE/.
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
STAMP="$(date +%Y%m%d-%H%M)"
EVID="EVIDENCE/run-${STAMP}"
mkdir -p "$EVID" out

echo "== model dir: $IP_ASR_MODEL_DIR ($(du -sh "$IP_ASR_MODEL_DIR" 2>/dev/null | cut -f1))"
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

echo "== transcribing 6 chunks (sampling TCP peers of the adapter process)"
IP_ASR_URL="http://127.0.0.1:${PORT}" IP_ASR_CONCURRENCY=2 IP_ASR_RETRIES=2 \
  .venv/bin/python -m asr_adapter.cli \
  --chunks data/chunks.json --out out/transcript.json \
  --hotwords "玥合冷链,青浦仓,温控箱,到货验收单,部分拒收单,返利点位,温控曲线,补冷媒,干线调度,季度对账单" \
  > "$EVID/adapter.log" 2>&1 &
CLI=$!

PEERS=()
while kill -0 "$CLI" 2>/dev/null; do
  while read -r peer; do PEERS+=("$peer"); done < <(ss -tnp 2>/dev/null | awk -v p="$CLI" '$0 ~ ("pid=" p ",") {print $5}')
  sleep 0.3
done
wait "$CLI" || echo "adapter exit=$?"
UNIQUE=$(printf '%s\n' "${PEERS[@]:-}" | grep -v '^$' | sort -u || true)
printf '%s\n' "${UNIQUE:-none captured}" > "$EVID/peers_adapter.txt"
echo "== adapter peers: ${UNIQUE:-none captured}"

cp out/transcript.json "$EVID/transcript.json"
echo "== scoring"
.venv/bin/python eval/compute_cer.py --transcript out/transcript.json \
  --truth eval/fixtures/ground_truth.json --out "$EVID/cer_report.md" | tee "$EVID/cer_stdout.txt"

echo "== egress check (client-side stub run, same assertion the review asks for)"
OUT="$EVID/egress_check.json" bash scripts/verify_no_egress.sh > "$EVID/egress_check.txt" 2>&1 || true
tail -2 "$EVID/egress_check.txt"

echo "== service log tail"
tail -5 "$EVID/service.log"
echo "== artefacts in $EVID"
ls -la "$EVID"
