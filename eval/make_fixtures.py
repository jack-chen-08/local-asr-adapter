"""Build the committed test fixture: a fictional phone call, synthesised once.

Why synthesised: the ticket forbids real call recordings, so the fixture is a scripted
conversation read by two TTS voices. The text is invented, the audio contains no human
speaker, and the resulting WAV is committed so every later run needs no network at all.

    python eval/make_fixtures.py          # writes eval/fixtures/*.wav + ground_truth.json
"""
from __future__ import annotations

import asyncio
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
GAP_MS = 320
VOICES = {"客户": "zh-CN-YunxiNeural", "客服": "zh-CN-XiaoxiaoNeural"}

# Fictional entities only: no real company, product or person appears here.
DIALOGUE: list[tuple[str, str]] = [
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
]

HOTWORDS = ["玥合冷链", "青浦仓", "温控箱", "到货验收单", "部分拒收单", "返利点位",
            "温控曲线", "补冷媒", "干线调度", "季度对账单"]


def synth(text: str, voice: str, dest: Path) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        mp3 = Path(tmp.name)
    subprocess.run(
        [sys.executable, "-m", "edge_tts", "--voice", voice, "--text", text,
         "--write-media", str(mp3)],
        check=True, capture_output=True,
    )
    audio, sr = sf.read(mp3)
    mp3.unlink(missing_ok=True)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SR)
    return audio.astype(np.float32)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pieces: list[np.ndarray] = []
    truth: list[dict] = []
    cursor = 0
    for speaker, text in DIALOGUE:
        audio = synth(text, VOICES[speaker], OUT / "turn.mp3")
        start = cursor
        pieces.append(audio)
        cursor += int(len(audio) / SR * 1000)
        truth.append({"speaker": speaker, "start_ms": start, "end_ms": cursor, "text": text})
        pieces.append(np.zeros(int(SR * GAP_MS / 1000), dtype=np.float32))
        cursor += GAP_MS

    full = np.concatenate(pieces)
    wav = OUT / "dialogue.wav"
    sf.write(wav, full, SR, subtype="PCM_16")
    (OUT / "ground_truth.json").write_text(
        json.dumps(
            {
                "audio": wav.name,
                "sample_rate": SR,
                "duration_ms": int(len(full) / SR * 1000),
                "synthetic": True,
                "tts_voices": VOICES,
                "hotwords": HOTWORDS,
                "segments": truth,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{wav}  {len(full)/SR:.1f}s  {wav.stat().st_size/1e6:.1f}MB  turns={len(truth)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
