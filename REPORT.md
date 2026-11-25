# 验收报告 · 语音转写引擎适配层（#2284）

> 中文正文，命令与变量名英文。所有数字来自 `EVIDENCE/run-*/` 中的原始产物，可逐条复核。

## 1. 交付物

| 文件 | 作用 |
|---|---|
| `src/asr_adapter/` | 适配层：配置、HTTP 客户端、重叠去重、并发管线、CLI（退出码 0/2/3） |
| `service/server.py` | 本地推理服务：FunASR Paraformer-zh + fsmn-vad + ct-punc-c，`/healthz`、`/transcribe` |
| `Dockerfile.asr` | 构建期烘焙模型（953M + 3.9M + 283M），运行期无需外网 |
| `Dockerfile.adapter` | 纯客户端镜像，无模型、无 torch |
| `docker-compose.yml` | 两个服务，`asr` 不发布端口，仅内网可达 |
| `scripts/split_audio.py` | 30s 切块 + 2s 重叠，产出 `chunks.json` |
| `scripts/run_real_eval.sh` | 一键端到端：起服务 → 转写 → 评分 → 断网采样 |
| `scripts/verify_no_egress.sh` | 客户端外联自证（非回环对端即失败） |
| `eval/make_fixtures.py` · `eval/compute_cer.py` | 夹具再生、CER/热词/RTF 评分 |
| `tests/` | 17 项单测（含去重 3 项、日志不落内容 1 项） |

## 2. 测试环境

| 项 | 值 |
|---|---|
| CPU | Intel i5-9300H @ 2.40GHz，8 逻辑核 |
| 内存 | 15 GB |
| Python | 3.12.13 |
| 推理栈 | torch 2.6.0+cpu · torchaudio 2.6.0+cpu · funasr 1.2.6 |
| 模型 | paraformer-zh v2.0.4（953M）· fsmn-vad（3.9M）· ct-punc-c（283M） |
| 设备 | CPU only，无 GPU |
| 并发 | 适配层 2 路并发请求同一服务（`IP_ASR_CONCURRENCY=2`） |

## 3. 夹具

- 时长 **142.9 s**，**23 轮**客服-客户对话，**544 个有效汉字**（去标点）
- 内容为虚构的冷链物流售后场景，全部由 TTS 合成（`zh-CN-YunxiNeural` / `zh-CN-XiaoxiaoNeural`），
  无真人录音、无真实企业名
- 10 个领域热词：玥合冷链、青浦仓、温控箱、到货验收单、部分拒收单、返利点位、温控曲线、补冷媒、干线调度、季度对账单
- 音频切成 6 块（30s/块，2s 重叠）

## 4. 指标

### 4.1 三轮实测（同一夹具、同一机器、逐轮改进）

| 轮次 | 配置 | 段数 | 去重 | **CER** | 热词召回 |
|---|---|---|---|---|---|
| `run-0912` | 仅 Paraformer，无 VAD | 6 | 0 | **7.54%** | 10/10 |
| `run-0935` | + fsmn-vad + ct-punc-c | 65 | 1 | **6.25%** | 10/10 |
| `run-0936` | + 去重判据修正 | 61 | **5** | **2.76%** | 10/10 |

`run-0912` 的 582 字假设 vs 544 字参考，多出的 38 字正是重叠区被重复转写的部分 —— 这是把
「只有一块一条整段、去重永远打不中」暴露出来的原因，也是 6.25% → 2.76% 的来源。

### 4.2 最终配置明细（`run-20261125-0936`）

| 指标 | 值 |
|---|---|
| CER（整体） | **2.76%** |
| 热词召回 | **100%（10/10）** |
| 失败块 | 0 / 6 |
| 块耗时 p50 / p95 / max | 7.54s / 15.11s / 18.38s（含首次模型加载） |
| 墙钟 | 29.78s（142.9s 音频） |
| **RTF（墙钟）** | **0.208** |
| RTF（块耗时求和） | 0.392 |

RTF 墙钟 0.208 表示 **1 小时录音约需 12.5 分钟**（2 路并发、8 核 CPU）。ptt 单块冷启动
（模型加载 5.5s）会抬高 p95，长批量任务中该成本被摊薄。

## 5. 断网自证

两条独立证据，产物在 `EVIDENCE/run-20261125-0936/`：

1. **真实运行期采样**：转写全程每 0.3s 采样适配层进程的 TCP 对端，结果
   `peers_adapter.txt` 只有 `127.0.0.1:8809` —— 只与本地服务通信。
2. **客户端自证**：`verify_no_egress.sh` 用桩服务跑同一份 `chunks.json`，
   任何非回环对端即 `exit 1`；本次输出 `PASS: no non-loopback peer observed`。

`docker run --network none` 形式在 `Dockerfile.asr` 中已具备条件（模型烘焙进镜像），
需在有 Docker 的环境复核；本机为 WSL + 无 GPU，未构建镜像。

## 6. 复现

```bash
# 1) 依赖（venv 内）
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0+cpu" "torchaudio==2.6.0+cpu"
pip install -r requirements-service.txt && pip install -e ".[dev]"

# 2) 取模型（构建期一次性，等价于 Dockerfile.asr 的 bake 步骤）
python scripts/fetch_model.py           # 或使用 MODELSCOPE_CACHE 预置的本地目录

# 3) 端到端 + 评分，产物落 EVIDENCE/run-<stamp>/
bash scripts/run_real_eval.sh

# 4) 单测
pytest -q
```

## 7. 已知限制（不影响验收项，如实列出）

| 限制 | 影响 | 处理 |
|---|---|---|
| 夹具是 TTS 合成音 | 真人口音/嘈杂环境下 CER 会高于 2.76% | 客户可用自有录音替换 `eval/fixtures/`，脚本不变 |
| 未做说话人分离 | 转写不带 speaker 标签 | 需求未要求；`fsmn-vad` 已预留分段基础 |
| 本地无 ffmpeg | 非 wav 输入需转码 | 镜像内已装 ffmpeg；本地验证只喂 wav |
| 未构建 Docker 镜像 | `--network none` 未在本机复现 | Dockerfile 已按烘焙模型写好，需有 Docker 的环境验一次 |
| 单服务单 worker | 并发上限受 CPU 核数约束 | 压测 RTF 0.208 已满足 1 小时录音 15 分钟内出稿 |
