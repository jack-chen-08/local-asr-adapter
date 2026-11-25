# Chunked ASR adapter (in-cloud, no egress)

A thin, auditable layer that turns one long recording into a reviewable transcript draft by
calling a **local** speech-recognition service. The adapter is deliberately boring: one HTTP
call per chunk, millisecond timestamps on the global timeline, retries, overlap de-duplication
and a failure list a human can act on.

中文提要：本仓库是「本地化 ASR 适配层」。模型与推理只跑在客户自有云内的 `asr` 容器里（构建期烘焙模型，
运行时断网可用）；适配层只做切块调用、全局毫秒时间戳换算、重叠去重、重试与失败清单，不接触任何第三方
在线转写接口，也不把音频或识别文本写进日志。

## Layout

| path | purpose（用途） |
|---|---|
| `src/asr_adapter/config.py` | env-only settings; a missing `IP_ASR_URL` fails loudly（缺地址直接报错，不静默回退公网） |
| `src/asr_adapter/client.py` | one HTTP call per chunk, retry policy, local→global offset |
| `src/asr_adapter/dedupe.py` | drops the duplicate copy of a sentence in the overlap region |
| `src/asr_adapter/pipeline.py` | bounded concurrency, ordered assembly, per-chunk timing, partial-failure report |
| `src/asr_adapter/cli.py` | end-to-end entry point, exit codes 0 / 2 / 3 |
| `service/server.py` | FastAPI wrapper around FunASR Paraformer, `/healthz` + `/transcribe` |
| `Dockerfile.asr` | bakes the model at build time so runtime needs no network |
| `Dockerfile.adapter` | client-only image |
| `scripts/split_audio.py` | cuts a WAV into overlapping chunks + `chunks.json` |
| `eval/make_fixtures.py` | regenerates the synthetic fixture (TTS, fictional dialogue) |
| `eval/compute_cer.py` | CER, per-turn CER, hotword recall, RTF |

## Quick start

```bash
docker compose up -d asr                       # load/health gate; model already inside
python scripts/split_audio.py call.wav data --chunk 30 --overlap 2
IP_ASR_URL=http://127.0.0.1:8809 IP_ASR_CONCURRENCY=2 \
  python -m asr_adapter.cli --chunks data/chunks.json --out out/transcript.json \
                            --hotwords "玥合冷链,青浦仓,温控箱"
```

Exit codes: `0` every chunk transcribed, `2` configuration/service-unhealthy, `3` partial —
the draft is still written and `errors[]` names the chunks to retry.

## Configuration

| env | default | meaning |
|---|---|---|
| `IP_ASR_URL` | **required** | base URL of the local service, e.g. `http://127.0.0.1:8809` |
| `IP_ASR_CONCURRENCY` | `2` | in-flight chunks (must stay ≤ what the service can hold in memory) |
| `IP_ASR_RETRIES` | `2` | retries per chunk on network error / 429 / 5xx |
| `IP_ASR_TIMEOUT_S` | `120` | per-request timeout |
| `IP_ASR_SAMPLE_RATE` | `16000` | declared to the service; the splitter normalises audio |

## Service contract

```
GET  {IP_ASR_URL}/healthz    -> {"status":"ok","model":"paraformer-zh","model_loaded":true}
POST {IP_ASR_URL}/transcribe multipart/form-data
     file=@chunk.wav  hotwords='["玥合冷链"]'  sample_rate=16000
  -> {"segments":[{"start_ms":1000,"end_ms":2400,"text":"...","confidence":0.93}]}
```

Times returned by the service are **chunk-relative**; the adapter adds the chunk offset, so
every timestamp downstream is global milliseconds.（服务返回块内时间，适配层统一加上块起点。）

## Design decisions

**Overlap, then de-duplicate.** Chunks overlap by 2 s so no word is cut in half; the repeated
region is removed by *time overlap plus text similarity* (`difflib` ratio ≥ 0.80 after
punctuation stripping), keeping the longer copy. Time-only matching would delete genuine
repeats ("好的好的"); text-only matching would delete repeated stock phrases spoken minutes
apart. The removal count is reported, never silent.

**A bad chunk must not sink the batch.** After `IP_ASR_RETRIES` the chunk is recorded in
`errors[]` with a reason and the rest of the transcript is still produced (exit `3`), because
the review queue is cheaper than a re-run of the whole call. 4xx responses from our own
service are *not* retried — retrying cannot fix a contract problem.

**Privacy by construction.** Audio bytes and recognised text never reach a logger: the
service logs byte counts and segment counts, the pipeline logs chunk ids and counts. A
regression test asserts a spoken sentence is absent from every captured log record. Nothing
outside `IP_ASR_URL` is ever contacted; another test fails if any request URL leaves that host.

## Verification

```bash
python -m pytest -q            # 14 tests, no model and no network required
```

Measured results (CER, hotword recall, RTF, air-gapped run) are recorded in
`EVIDENCE/` and summarised in `REPORT.md`.
