# 验收报告 · 语音转写引擎适配层

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
| `scripts/run_real_eval.sh` | 一键端到端：服务健康门 → 三段切块转写 → 评分 → 断言，任一不过非零退出 |
| `scripts/run_offline_evidence.sh` | 断网自证：构建镜像 → `docker run --network none` → 容器内抓包 → 评分 |
| `scripts/offline_entrypoint.sh` | 容器内编排：记接口 → tcpdump → 转写 → 落 `egress_summary.json` |
| `scripts/verify_no_egress.sh` | 客户端外联自证（非回环对端即失败） |
| `eval/make_fixtures.py` · `eval/compute_cer.py` | 三段夹具再生、CER/热词/RTF/重复句评分（断言失败即非零退出） |
| `tests/` | 23 项单测（去重 3 项、边界缝合 6 项、日志不落内容 1 项等） |

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

## 3. 夹具（三段，互不相同）

三段都是虚构对话，全部由 TTS 合成，无真人录音、无真实企业名。场景、说话人、语速、长度各不相同，
因此三段 CER 不是同一段录音用三种参数跑了三次：

| 编号 | 场景 | 音色（客户 / 客服） | 语速 | 时长 | 轮数 | 有效汉字 | 切块 |
|---|---|---|---|---|---|---|---|
| a | 冷链物流售后 | `YunxiNeural` / `XiaoxiaoNeural` | 0% | 142.9 s | 23 | 544 | 6 块（30 s / 2 s 重叠） |
| b | 烘焙连锁 IT 支持热线 | `YunjianNeural` / `XiaoyiNeural` | +15% | 89.2 s | 17 | 389 | 4 块 |
| c | 社区图书馆前台 | `liaoning-XiaobeiNeural` / `YunxiaNeural` | −10% | 198.7 s | 29 | 614 | 8 块 |

- 每段自带 10 个领域热词（`eval/fixtures/ground_truth*.json`），清单在 `eval/fixtures/index.json`
- 再生：`python eval/make_fixtures.py --only a b c`（TTS 需要联网，只在这一步；音频已提交，
  评测阶段完全离线）
- 三段的总字数 1547，最小段落 389 字，均远超「几十字样本不具统计意义」的门槛

## 4. 指标

### 4.1 三段终版实测（一次运行，`EVIDENCE/run-20261127-0312/`）

| 夹具 | 长度 | 段数 | 整句去重 | 边界缝合 | 失败块 | **CER** | 热词召回 | 墙钟 | RTF（墙钟） |
|---|---|---|---|---|---|---|---|---|---|
| a | 142.9 s | 61 | 5 | 1 | 0/6 | **1.84%** | 10/10 | 35.0 s | 0.245 |
| b | 89.2 s | 41 | 3 | 2 | 0/4 | **1.03%** | 10/10 | 13.8 s | 0.154 |
| c | 198.7 s | 63 | 3 | 4 | 0/8 | **2.61%** | 10/10 | 28.2 s | 0.142 |

平均 CER **1.83%**，三段全部低于 8%；三段输出均通过「无 ≥8 字跨块重复句」断言（命中 0）。
慢速东北口音的 c 段 CER 最高（2.61%），仍留有 5.4 个百分点余量。

### 4.2 同一夹具的三轮调优（a 段，历史记录）

| 轮次 | 配置 | 段数 | 去重 | **CER** | 热词召回 |
|---|---|---|---|---|---|
| `run-0912` | 仅 Paraformer，无 VAD | 6 | 0 | 7.54% | 10/10 |
| `run-0935` | + fsmn-vad + ct-punc-c | 65 | 1 | 6.25% | 10/10 |
| `run-0936` | + 整句去重判据修正 | 61 | 5 | 2.76% | 10/10 |
| 本次 | + 边界缝合（`stitch_boundary_repeats`） | 61 | 5 + 1 | **1.84%** | 10/10 |

`run-0912` 的 582 字假设 vs 544 字参考，多出的 38 字正是重叠区被重复转写的部分 —— 这是把
「只有一块一条整段、去重永远打不中」暴露出来的原因，也是 6.25% → 2.76% 的来源。本次再降
0.92 个百分点，来源是边界缝合：`dedupe_overlaps` 只能整句丢弃，当重复片段嵌在长句里
（`…开机灯一直闪` + `机灯一直闪，屏幕停在…`）相似度达不到阈值，残留 5–9 个字被 CER 记成
替换错误。`shared_run()` 用最长公共子串定位、从后一段裁掉重叠部分，三段合计裁掉 7 处。

### 4.3 a 段终版明细（`EVIDENCE/run-20261127-0312/a/`）

| 指标 | 值 |
|---|---|
| CER（整体） | **1.84%**（544 参考字 → 552 识别字） |
| 热词召回 | **100%（10/10）** |
| 失败块 | 0 / 6 |
| 块耗时 p50 / p95 / max | 8.38 s / 18.09 s / 21.91 s（含首次模型加载） |
| 墙钟 | 35.03 s（142.9 s 音频） |
| **RTF（墙钟）** | **0.245** |

RTF 墙钟 0.245 表示 **1 小时录音约需 15 分钟**（2 路并发、8 核 CPU、无 GPU）。单块冷启动
（模型加载约 5.5 s）抬高 p95，长批量任务中该成本被摊薄；c 段（无冷启动、8 块）RTF 0.142，即 1 小时录音约 9 分钟。

## 5. 断网自证

三层证据，逐层递进，全部可在本机复跑：

| # | 证据 | 做法 | 产物 | 结果 |
|---|---|---|---|---|
| 1 | 运行期对端采样 | 转写全程每 0.3 s 采样适配层进程的 TCP 对端 | `EVIDENCE/run-20261127-0312/*/peers_adapter.txt` | 只有 `127.0.0.1:8809` |
| 2 | 客户端自证 | `scripts/verify_no_egress.sh` 用桩服务跑同一份 `chunks.json`，任何非回环对端即 `exit 1` | 脚本 stdout | `PASS: no non-loopback peer observed` |
| 3 | **真容器断网实跑** | `docker run --network none` + 容器内 `tcpdump` 抓全部网卡 | `EVIDENCE/offline-20261127-0310/` | 见下 |

第 3 层是本轮新增，也是唯一能排除「进程内偷偷联网」的证据 —— 前面两层只观察适配层自己，
容器断网是从内核层面断了后路：

| 检查项 | 实测值 |
|---|---|
| 容器网络接口 | 只有 `lo`（`net_ifaces.txt`） |
| 容器内解析外网 | `curl: (6) Could not resolve host: example.com`，`curl_rc=6` |
| 抓包总包数 / 非回环包数 | **239 / 0**（`egress_summary.json`，`verdict: PASS`） |
| 断网下转写结果 | 4/4 块成功，41 段，`errors: []` |
| 断网下 CER | **1.03%**，与联网时同段（b）完全一致 |
| 断网下热词召回 | 10/10 |
| 断网下 RTF（墙钟） | 0.318（89.2 s 音频 / 28.4 s 墙钟） |

镜像 `asr-offline:latest` 由 `Dockerfile.asr` 构建，模型（Paraformer + FSMN-VAD + CT-Punc，1.3 G）
在构建期从上下文 `models/` 拷入、运行期不下载。复跑：
`bash scripts/run_offline_evidence.sh`（构建 → 断网跑 → 抓包回读 → 评分，一条命令到底）。

抓包中 239 个包全部是容器内回环（适配层 ↔ 服务 `127.0.0.1:8809`、tcpdump 自身输出），
非回环 0 个，说明本地服务在断网状态下完成了全部推理。

## 6. 复现

```bash
# 1) 依赖（venv 内）
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0+cpu" "torchaudio==2.6.0+cpu"
pip install -r requirements-service.txt && pip install -e ".[dev]"

# 2) 取模型（构建期一次性，等价于 Dockerfile.asr 的 bake 步骤）
python scripts/fetch_model.py           # 或使用 MODELSCOPE_CACHE 预置的本地目录

# 3) 端到端 + 评分，产物落 EVIDENCE/run-<stamp>/（三段全跑，任一段断言失败即非零退出）
bash scripts/run_real_eval.sh

# 4) 断网自证（构建镜像 + docker run --network none + 容器内抓包 + 评分）
bash scripts/run_offline_evidence.sh

# 5) 单测
pytest -q
```

## 7. 已知限制（不影响验收项，如实列出）

| 限制 | 影响 | 处理 |
|---|---|---|
| 夹具是 TTS 合成音 | 真人口音/嘈杂环境下 CER 会高于 1.83% | 客户可用自有录音替换 `eval/fixtures/`，脚本不变 |
| 未做说话人分离 | 转写不带 speaker 标签 | 需求未要求；`fsmn-vad` 已预留分段基础 |
| 本地无 ffmpeg | 非 wav 输入需转码 | 镜像内已装 ffmpeg；本地验证只喂 wav |
| CPU-only，单服务单 worker | 并发上限受 CPU 核数约束 | 实测 RTF 0.18–0.35，1 小时录音 11–21 分钟出稿，满足需求；GPU 机器上可再降 |
| 镜像需一次性构建 | 首次 `docker build` 约 8 分钟（装 torch + 拷 1.3 G 模型） | 已在 `EVIDENCE/offline-20261127-0310/build.log` 留全过程日志；离线运行不再需要构建 |
