#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def audio_info(path: Path):
    info = sf.info(str(path))
    return {
        "frames": int(info.frames),
        "sample_rate": int(info.samplerate),
        "seconds": float(info.frames) / float(info.samplerate),
    }


def trailing_low_energy_seconds(path: Path, frame_seconds: float = 0.05, threshold_db: float = -35.0) -> float:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size == 0:
        return 0.0

    hop = max(1, int(sr * frame_seconds))
    frame_rms = []
    for start in range(0, len(wav), hop):
        chunk = wav[start : start + hop]
        if chunk.size == 0:
            continue
        frame_rms.append(float(np.sqrt(np.mean(np.square(chunk))) + 1e-12))
    if not frame_rms:
        return 0.0

    peak_rms = max(frame_rms)
    cutoff = peak_rms * (10.0 ** (threshold_db / 20.0))
    trailing = 0
    for rms in reversed(frame_rms):
        if rms <= cutoff:
            trailing += 1
        else:
            break
    return round(trailing * frame_seconds, 6)


def resolve_target_audio(row: dict, raw_audio_root: Path | None):
    audio_path = Path(row["audio"])
    if raw_audio_root is not None:
        preferred = raw_audio_root / f"{audio_path.stem}.wav"
        if preferred.exists():
            return preferred
    return audio_path


def normalize_sample_id(row: dict, index: int, fallback_stem: str = ""):
    if row.get("sample_id"):
        return str(row["sample_id"])
    if row.get("audio"):
        return Path(row["audio"]).stem
    if fallback_stem:
        return fallback_stem
    return f"{index:02d}"


def enrich_eval_rows(eval_rows, reference_rows):
    reference_by_text = {row.get("text", ""): row for row in reference_rows}
    normalized = {}
    for index, row in enumerate(eval_rows):
        enriched = dict(row)
        if ("audio" not in enriched or "ref_audio" not in enriched) and enriched.get("text") in reference_by_text:
            ref = reference_by_text[enriched["text"]]
            enriched.setdefault("audio", ref.get("audio", ""))
            enriched.setdefault("ref_audio", ref.get("ref_audio", ""))
        sample_id = normalize_sample_id(enriched, index)
        enriched["sample_id"] = sample_id
        normalized[sample_id] = enriched
    return normalized


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed_eval_dir", type=str, required=True, help="checkpoint fixed_eval_audio dir")
    parser.add_argument("--fixed_eval_jsonl", type=str, default=None)
    parser.add_argument("--reference_jsonl", type=str, default=None)
    parser.add_argument("--raw_audio_root", type=str, default="assets/BZNSYP/Wave")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    fixed_eval_dir = Path(args.fixed_eval_dir)
    manifest_path = fixed_eval_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    samples = manifest.get("samples", [])

    fixed_eval_jsonl = Path(args.fixed_eval_jsonl) if args.fixed_eval_jsonl else None
    eval_rows = {}
    eval_rows_by_text = {}
    if fixed_eval_jsonl is not None and fixed_eval_jsonl.exists():
        reference_rows = []
        if args.reference_jsonl and Path(args.reference_jsonl).exists():
            reference_rows = read_jsonl(Path(args.reference_jsonl))
        eval_rows = enrich_eval_rows(read_jsonl(fixed_eval_jsonl), reference_rows)
        eval_rows_by_text = {row.get("text", ""): row for row in eval_rows.values()}

    raw_audio_root = Path(args.raw_audio_root) if args.raw_audio_root else None
    if raw_audio_root is not None and not raw_audio_root.exists():
        raw_audio_root = None

    report_rows = []
    for index, row in enumerate(samples):
        sample_id = normalize_sample_id(row, index, fallback_stem=Path(row.get("wav_path", "")).stem)
        generated_path = Path(row["wav_path"])
        target_row = eval_rows.get(sample_id, row)
        if ("audio" not in target_row or not target_row.get("audio")) and row.get("text") in eval_rows_by_text:
            target_row = eval_rows_by_text[row["text"]]
        if "audio" not in target_row or not target_row.get("audio"):
            raise KeyError(
                f"Could not resolve target audio for sample_id={sample_id}. "
                "Pass a normalized fixed_eval_jsonl or add --reference_jsonl."
            )
        target_audio = resolve_target_audio(target_row, raw_audio_root)

        target_info = audio_info(target_audio)
        generated_info = audio_info(generated_path)
        duration_ratio = generated_info["seconds"] / max(target_info["seconds"], 1e-6)
        trailing_seconds = trailing_low_energy_seconds(generated_path)
        cap_hit = bool(row.get("cap_hit", False))
        inferred_anomaly = cap_hit or duration_ratio > 2.5 or generated_info["seconds"] <= 0.0
        report_rows.append(
            {
                "sample_id": sample_id,
                "text": row.get("text", ""),
                "target_audio": str(target_audio),
                "generated_audio": str(generated_path),
                "target_seconds": round(target_info["seconds"], 6),
                "generated_seconds": round(generated_info["seconds"], 6),
                "duration_ratio": round(duration_ratio, 6),
                "duration_abs_error": round(abs(generated_info["seconds"] - target_info["seconds"]), 6),
                "generated_tail_low_energy_seconds": trailing_seconds,
                "cap_hit": cap_hit,
                "stop_reason": row.get("stop_reason", ""),
                "is_anomaly": bool(row.get("is_anomaly", False)) or inferred_anomaly,
            }
        )

    report_rows.sort(
        key=lambda item: (
            float(item["duration_ratio"]),
            float(item["generated_tail_low_energy_seconds"]),
        ),
        reverse=True,
    )

    summary = {
        "fixed_eval_dir": str(fixed_eval_dir),
        "num_samples": len(report_rows),
        "num_anomalies": sum(1 for row in report_rows if row["is_anomaly"]),
        "max_duration_ratio": max((row["duration_ratio"] for row in report_rows), default=0.0),
        "max_tail_low_energy_seconds": max(
            (row["generated_tail_low_energy_seconds"] for row in report_rows), default=0.0
        ),
        "samples": report_rows,
    }

    output_json = Path(args.output_json) if args.output_json else fixed_eval_dir / "diagnosis_report.json"
    output_csv = Path(args.output_csv) if args.output_csv else fixed_eval_dir / "diagnosis_report.csv"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_csv(output_csv, report_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
