"""Build the committed test fixtures: fictional phone calls, synthesised once.

Why synthesised: the ticket forbids real call recordings, so each fixture is a scripted
conversation read by two TTS voices. The text is invented, the audio contains no human
speaker, and the resulting WAVs are committed so every later run needs no network at all.

Three scenarios differ in speakers, domain, speech rate and length, so the CER numbers
cannot all come from one recording:

  a  cold-chain logistics hotline   Yunxi + Xiaoxiao      normal rate   ~143 s   23 turns
  b  bakery-chain IT support line   Yunjian + Xiaoyi      +15% rate      ~85 s   17 turns
  c  community library front desk    Xiaobei + Yunxia      -10% rate     ~190 s   29 turns

    python eval/make_fixtures.py --only b c      # build the missing scenarios
    python eval/make_fixtures.py --only a        # rebuild scenario a (dialogue.wav)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

OUT = Path(__file__).resolve().parent / "fixtures"
SR = 16_000

SCENARIOS: dict[str, dict] = {
    "a": {
        "wav": "dialogue.wav",
        "truth": "ground_truth.json",
        "gap_ms": 320,
        "rate": None,
        "voices": {"客户": "zh-CN-YunxiNeural", "客服": "zh-CN-XiaoxiaoNeural"},
        "hotwords": ["玥合冷链", "青浦仓", "温控箱", "到货验收单", "部分拒收单", "返利点位",
                     "温控曲线", "补冷媒", "干线调度", "季度对账单"],
        "turns": [
            ("客服", "您好，玥合冷链客服中心，请问有什么可以帮您。"),
            ("客户", "你好，我这边是青浦仓的收货组，昨天到的那批货温度好像不对。"),
            ("客服", "您说的是昨天下午到青浦仓的三号车吗，车牌尾号是七三二那一趟。"),
            ("客户", "对，就是那趟，我们抽检的时候发现三号温控箱的记录偏高了两度。"),
            ("客服", "明白，我先把这张到货验收单调出来，您方便报一下单号吗。"),
            ("客户", "单号是 K F 二零七六，后面还有四位是三九一八。"),
            ("客服", "好的，KF 二零七六三九一八，我看到这趟车在途中补过一次冷媒。"),
            ("客户", "补冷媒我们是知道的，但是补完之后温控曲线没有回落到零下十八度。"),
            ("客服", "这个情况我需要升级给干线调度，按合同第六条属于异常到货，可以拒收。"),
            ("客户", "我们不想整批拒收，只想把这三箱单独挑出来做退货处理。"),
            ("客服", "可以，那我给您开一张部分拒收单，剩下的货正常入库，不影响到货时效。"),
            ("客户", "行，那退运的运费怎么算，是我们承担还是你们承担。"),
            ("客服", "因为是温控异常，按合约由承运方承担，运费会走月度对账抵扣。"),
            ("客户", "好，另外上个季度的返利点位是不是也该结算了。"),
            ("客服", "返利点位按百分之一点五算，季度对账单我这边明天发您邮箱。"),
            ("客户", "邮箱还是发货通知那个，收件人是财务的小周。"),
            ("客服", "收到，我会同时抄送青浦仓的负责人，麻烦您留意一下附件。"),
            ("客户", "还有一件事，下个月的温控箱能不能多配两台 K F 二零。"),
            ("客服", "KF 二零我这边看到还有十二台可用，您需要几台，我帮您预留。"),
            ("客户", "先留四台，等排产确认了我再追加。"),
            ("客服", "好的，四台我记下来，预留到本周五，过期自动释放。"),
            ("客户", "可以，那就这样，退货单出来麻烦发我一份。"),
            ("客服", "没问题，退货单和部分拒收单我今天下午一起发您，感谢来电。"),
        ],
    },
    "b": {
        "wav": "dialogue_b.wav",
        "truth": "ground_truth_b.json",
        "gap_ms": 260,
        "rate": "+15%",
        "voices": {"客户": "zh-CN-YunjianNeural", "客服": "zh-CN-XiaoyiNeural"},
        "hotwords": ["收银机", "小票打印机", "巡检工单", "备用机", "上门服务单",
                     "故障代码", "电源适配器", "月度巡检报告", "年度维保", "门店编号"],
        "turns": [
            ("客服", "您好，欣野烘焙门店支持热线，请问设备出了什么问题。"),
            ("客户", "我这里是星港路店的店长，收银机今天早上开不了机。"),
            ("客服", "麻烦报一下门店编号，我看一下这台机器的巡检工单记录。"),
            ("客户", "门店编号是 C Y 三四七，机器上贴的资产标签尾号是九零二。"),
            ("客服", "C Y 三四七，资产尾号九零二，最后一次巡检是上周二，状态正常。"),
            ("客户", "现在开机灯一直闪，屏幕停在启动画面，小票打印机也没有反应。"),
            ("客服", "您先把电源适配器换到备用机那条线试试，看看是不是供电不稳。"),
            ("客户", "换过了，还是老样子，备用机能不能先送过来顶一天。"),
            ("客户", "早高峰马上开始，没有收银机只能手写小票。"),
            ("客服", "我给您开一张加急上门服务单，工程师今天十点前到店。"),
            ("客服", "备用机从区域仓调拨，随工程师一起带过去。"),
            ("客户", "那我这边需要准备什么，旧的机器要不要留着。"),
            ("客服", "旧机器请留着，工程师要读取故障代码，方便判断是不是主板问题。"),
            ("客户", "好，配件要不要另外收费，我们签的是年度维保。"),
            ("客服", "年度维保内不收上门费和工时费，只有主板这种大件按成本价。"),
            ("客服", "上门服务单号我发到店长的手机上，工单结果会写进月度巡检报告。"),
            ("客户", "行，我在店里等，麻烦让工程师到了直接找我。"),
        ],
    },
    "c": {
        "wav": "dialogue_c.wav",
        "truth": "ground_truth_c.json",
        "gap_ms": 420,
        "rate": "-10%",
        "voices": {"客户": "zh-CN-liaoning-XiaobeiNeural", "客服": "zh-CN-YunxiaNeural"},
        "hotwords": ["多功能厅", "读者证", "借阅证", "逾期罚金", "自助还书机",
                     "馆藏检索", "续借", "场地占用承诺书", "通借通还", "集体借阅"],
        "turns": [
            ("客服", "您好，临河社区图书馆，请问需要什么帮助。"),
            ("客户", "我想问一下，多功能厅这个周末还能不能预约。"),
            ("客服", "您是给单位活动预约，还是个人读书会用，两种情况流程不一样。"),
            ("客户", "是我们读书小组，一共八个人，周六下午两点到四点。"),
            ("客服", "周六下午多功能厅已经有一场亲子手工，下午三点半结束。"),
            ("客户", "那能不能从三点半开始，我们延到五点半。"),
            ("客服", "可以，三点半到五点半两小时，中间留十分钟清场，需要您的读者证号。"),
            ("客户", "读者证号我记不清了，是用手机号办的那张。"),
            ("客户", "手机号尾号是二零八八，姓名的第二个字是慧。"),
            ("客服", "查到您的借阅证了，还有三本书在借，其中一本已经逾期四天。"),
            ("客户", "逾期那本我今天下午拿过来还，罚金怎么算。"),
            ("客服", "逾期罚金每天两毛，四天是八毛，还书时在自助还书机上直接结清。"),
            ("客户", "那续借怎么弄，另外两本我还想再看一周。"),
            ("客服", "续借在馆藏检索里点续借，一次七天，最多续两次，被预约的书不能续。"),
            ("客户", "明白了，场地预约要交押金吗，能不能开发票。"),
            ("客服", "公益读书会免押金，只需要签一份场地占用承诺书。"),
            ("客服", "发票可以开，抬头写单位名称，活动结束三个工作日发电子票。"),
            ("客户", "那我把八个人的名单发到哪个邮箱。"),
            ("客服", "发到馆里的公共邮箱，主题写场地预约加日期，我这边收到就登记。"),
            ("客户", "多功能厅里有投影和音响吗，我们自己带了笔记本。"),
            ("客服", "有投影和无线话筒，接口是高清线，音响需要提前一天调试。"),
            ("客户", "那周五能不能先去看一下设备。"),
            ("客服", "可以，周五上午十点我在前台等您，带您走一遍设备。"),
            ("客户", "还有一件事，我们想借一批同主题的书，大概二十本。"),
            ("客服", "同主题二十本可以走通借通还的集体借阅，期限三十天。"),
            ("客户", "集体借阅需要单位证明吗。"),
            ("客服", "需要一份读书小组的登记表和负责人身份证复印件，其他不需要。"),
            ("客户", "好，我明天上午把材料送过来，顺便把逾期的书还了。"),
            ("客服", "好的，我把场地和集体借阅都先占位，材料到了再确认。"),
        ],
    },
}


def synth(text: str, voice: str, rate: str | None) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        mp3 = Path(tmp.name)
    cmd = [sys.executable, "-m", "edge_tts", "--voice", voice, "--text", text,
           "--write-media", str(mp3)]
    if rate:
        cmd.append("--rate=" + rate)  # equals form: argparse would read "-10%" as a flag
    subprocess.run(cmd, check=True, capture_output=True)
    audio, sr = sf.read(mp3)
    mp3.unlink(missing_ok=True)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SR)
    return audio.astype(np.float32)


def build(key: str) -> dict:
    sc = SCENARIOS[key]
    OUT.mkdir(parents=True, exist_ok=True)
    pieces: list[np.ndarray] = []
    truth: list[dict] = []
    cursor = 0
    for speaker, text in sc["turns"]:
        audio = synth(text, sc["voices"][speaker], sc["rate"])
        start = cursor
        pieces.append(audio)
        cursor += int(len(audio) / SR * 1000)
        truth.append({"speaker": speaker, "start_ms": start, "end_ms": cursor, "text": text})
        gap = np.zeros(int(SR * sc["gap_ms"] / 1000), dtype=np.float32)
        pieces.append(gap)
        cursor += sc["gap_ms"]

    full = np.concatenate(pieces)
    wav = OUT / sc["wav"]
    sf.write(wav, full, SR, subtype="PCM_16")
    doc = {
        "audio": wav.name,
        "scenario": key,
        "sample_rate": SR,
        "duration_ms": int(len(full) / SR * 1000),
        "synthetic": True,
        "tts_voices": sc["voices"],
        "tts_rate": sc["rate"] or "0%",
        "hotwords": sc["hotwords"],
        "segments": truth,
    }
    (OUT / sc["truth"]).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{wav}  {len(full)/SR:.1f}s  {wav.stat().st_size/1e6:.1f}MB  turns={len(truth)}")
    return doc


def write_index() -> None:
    rows = []
    for key, sc in SCENARIOS.items():
        wav, truth = OUT / sc["wav"], OUT / sc["truth"]
        if not (wav.exists() and truth.exists()):
            continue
        meta = json.loads(truth.read_text(encoding="utf-8"))
        rows.append({
            "id": key,
            "audio": wav.name,
            "truth": truth.name,
            "duration_ms": meta["duration_ms"],
            "turns": len(meta["segments"]),
            "voices": meta.get("tts_voices"),
            "rate": meta.get("tts_rate", "0%"),
            "hotwords": meta["hotwords"],
        })
    (OUT / "index.json").write_text(
        json.dumps({"max_cer": 0.08, "fixtures": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"index -> {OUT/'index.json'}  ({len(rows)} fixtures)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(SCENARIOS), default=None,
                    help="build only these scenarios (default: any whose wav is missing)")
    args = ap.parse_args()
    keys = args.only or [k for k, sc in SCENARIOS.items() if not (OUT / sc["wav"]).exists()]
    if not keys:
        print("all fixtures present; use --only to rebuild")
        write_index()
        return 0
    for k in keys:
        build(k)
    write_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
