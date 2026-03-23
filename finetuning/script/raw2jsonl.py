#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 BZNSYP / DataBaker 自动生成 Qwen3-TTS 微调所需 JSONL。
已兼容：
- 音频目录单独放在 assets/BZNSYP_24k/Wave
- 文本目录仍在 assets/BZNSYP/ProsodyLabeling

输出：
- train_raw.jsonl
- val_raw.jsonl
- test_raw.jsonl
- meta.json

Qwen3-TTS 官方输入格式：
{"audio":"...wav","text":"...","ref_audio":".../ref.wav"}
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple


PROSODY_TAG_RE = re.compile(r"#\d+")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """去掉 #1/#2/#3/#4 韵律标签，保留中文标点。"""
    text = text.strip().replace("\ufeff", "")
    text = PROSODY_TAG_RE.sub("", text)
    text = MULTI_SPACE_RE.sub("", text)
    return text


def parse_prosody_file(txt_path: Path) -> Dict[str, str]:
    """
    解析 ProsodyLabeling/*.txt
    只读取形如：
        000001\t卡尔普#2陪外孙#1玩滑梯#4。
    的中文行，忽略下一行拼音。
    """
    mapping: Dict[str, str] = {}

    with txt_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue

            m = re.match(r"^(\d{6})\s*\t\s*(.+)$", line)
            if not m:
                continue

            utt_id = m.group(1)
            text = clean_text(m.group(2))
            if text:
                mapping[utt_id] = text

    return mapping


def collect_transcripts(prosody_dir: Path) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    txt_files = sorted(prosody_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"在 {prosody_dir} 下没有找到 *.txt")

    for txt_file in txt_files:
        part = parse_prosody_file(txt_file)
        overlap = set(merged).intersection(part)
        if overlap:
            raise ValueError(f"发现重复 utt_id，示例: {sorted(list(overlap))[:5]}")
        merged.update(part)

    if not merged:
        raise ValueError(f"未能从 {prosody_dir} 解析出任何文本")
    return merged


def maybe_rel(path: Path, relative_to: Path | None) -> str:
    path = path.resolve()
    if relative_to is None:
        return str(path)
    return str(path.relative_to(relative_to.resolve()))


def build_records(
    audio_dir: Path,
    transcripts: Dict[str, str],
    ref_audio: Path,
    exclude_ref_utt_id: str | None = None,
    relative_to: Path | None = None,
) -> List[dict]:
    records: List[dict] = []

    for utt_id, text in sorted(transcripts.items()):
        if exclude_ref_utt_id and utt_id == exclude_ref_utt_id:
            continue

        wav_path = audio_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            continue

        records.append(
            {
                "audio": maybe_rel(wav_path, relative_to),
                "text": text,
                "ref_audio": maybe_rel(ref_audio, relative_to),
            }
        )

    return records


def split_records(
    records: List[dict],
    val_size: int,
    test_size: int,
    seed: int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    if val_size < 0 or test_size < 0:
        raise ValueError("val_size 和 test_size 不能为负数")
    if val_size + test_size >= len(records):
        raise ValueError("val_size + test_size 过大，导致 train 为空")

    items = records[:]
    rnd = random.Random(seed)
    rnd.shuffle(items)

    test_records = items[:test_size]
    val_records = items[test_size:test_size + val_size]
    train_records = items[test_size + val_size:]
    return train_records, val_records, test_records


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="重采样后的 wav 目录，例如 /home/.../assets/BZNSYP_24k/Wave",
    )
    parser.add_argument(
        "--prosody_dir",
        type=str,
        required=True,
        help="文本目录，例如 /home/.../assets/BZNSYP/ProsodyLabeling",
    )
    parser.add_argument(
        "--ref_audio",
        type=str,
        required=True,
        help="固定参考音频路径，例如 /home/.../assets/BZNSYP_24k/ref.wav",
    )
    parser.add_argument(
        "--exclude_ref_utt_id",
        type=str,
        default=None,
        help="若 ref.wav 来自某条原始句子，可填对应 6 位 utt_id，例如 000123",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="输出目录，例如 /home/.../Qwen3-TTS/ft_data",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=100,
        help="验证集条数，默认 100",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=100,
        help="测试集条数，默认 100",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42",
    )
    parser.add_argument(
        "--relative_to",
        type=str,
        default=None,
        help="若指定，则 JSONL 中路径写为相对此目录的相对路径",
    )

    args = parser.parse_args()

    audio_dir = Path(args.audio_dir).expanduser().resolve()
    prosody_dir = Path(args.prosody_dir).expanduser().resolve()
    ref_audio = Path(args.ref_audio).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    relative_to = Path(args.relative_to).expanduser().resolve() if args.relative_to else None

    if not audio_dir.exists():
        raise FileNotFoundError(f"audio_dir 不存在: {audio_dir}")
    if not prosody_dir.exists():
        raise FileNotFoundError(f"prosody_dir 不存在: {prosody_dir}")
    if not ref_audio.exists():
        raise FileNotFoundError(f"ref_audio 不存在: {ref_audio}")

    transcripts = collect_transcripts(prosody_dir)
    records = build_records(
        audio_dir=audio_dir,
        transcripts=transcripts,
        ref_audio=ref_audio,
        exclude_ref_utt_id=args.exclude_ref_utt_id,
        relative_to=relative_to,
    )

    if not records:
        raise ValueError("没有生成任何样本，请检查 audio_dir / prosody_dir / wav 文件名")

    train_records, val_records, test_records = split_records(
        records=records,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    write_jsonl(out_dir / "train_raw.jsonl", train_records)
    write_jsonl(out_dir / "val_raw.jsonl", val_records)
    write_jsonl(out_dir / "test_raw.jsonl", test_records)

    meta = {
        "total": len(records),
        "train": len(train_records),
        "val": len(val_records),
        "test": len(test_records),
        "audio_dir": str(audio_dir),
        "prosody_dir": str(prosody_dir),
        "ref_audio": maybe_rel(ref_audio, relative_to),
        "exclude_ref_utt_id": args.exclude_ref_utt_id,
        "seed": args.seed,
    }

    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("\n已输出：")
    print(out_dir / "train_raw.jsonl")
    print(out_dir / "val_raw.jsonl")
    print(out_dir / "test_raw.jsonl")
    print(out_dir / "meta.json")


if __name__ == "__main__":
    main()