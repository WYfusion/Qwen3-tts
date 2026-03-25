# coding=utf-8

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen_tts.eval_utils import summarize_audio_path

from .io_utils import read_json, read_jsonl, write_csv, write_json


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
        if ("audio" not in enriched or "ref_audio" not in enriched) and enriched.get("text", "") in reference_by_text:
            ref = reference_by_text[enriched["text"]]
            enriched.setdefault("audio", ref.get("audio", ""))
            enriched.setdefault("ref_audio", ref.get("ref_audio", ""))
        sample_id = normalize_sample_id(enriched, index)
        enriched["sample_id"] = sample_id
        normalized[sample_id] = enriched
    return normalized


def audio_metrics(path: Path):
    metrics = summarize_audio_path(path)
    metrics["seconds"] = float(metrics.get("decoded_seconds", 0.0))
    return metrics


def diagnose_eval_dir(
    *,
    fixed_eval_dir: Path,
    fixed_eval_jsonl: Path | None,
    reference_jsonl: Path | None,
    raw_audio_root: Path | None,
    duration_ratio_anomaly_threshold: float,
    peak_warn_threshold: float,
    clipped_frac_warn_threshold: float,
    hf_noise_warn_threshold: float,
    voiced_f0_delta_warn_threshold: float,
):
    manifest_path = fixed_eval_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    samples = manifest.get("samples", [])

    eval_rows = {}
    eval_rows_by_text = {}
    if fixed_eval_jsonl is not None and fixed_eval_jsonl.exists():
        reference_rows = read_jsonl(reference_jsonl) if reference_jsonl is not None and reference_jsonl.exists() else []
        eval_rows = enrich_eval_rows(read_jsonl(fixed_eval_jsonl), reference_rows)
        eval_rows_by_text = {row.get("text", ""): row for row in eval_rows.values()}

    if raw_audio_root is not None and not raw_audio_root.exists():
        raw_audio_root = None

    report_rows = []
    for index, row in enumerate(samples):
        sample_id = normalize_sample_id(row, index, fallback_stem=Path(row.get("wav_path", "")).stem)
        generated_path = Path(row["wav_path"])
        target_row = eval_rows.get(sample_id, row)
        if ("audio" not in target_row or not target_row.get("audio")) and row.get("text", "") in eval_rows_by_text:
            target_row = eval_rows_by_text[row["text"]]
        if "audio" not in target_row or not target_row.get("audio"):
            raise KeyError(
                f"Could not resolve target audio for sample_id={sample_id}. "
                "Pass a normalized fixed_eval_jsonl or add --reference_jsonl."
            )
        target_audio = resolve_target_audio(target_row, raw_audio_root)

        target_info = audio_metrics(target_audio)
        generated_info = audio_metrics(generated_path)
        duration_ratio = generated_info["seconds"] / max(target_info["seconds"], 1e-6)
        trailing_seconds = float(generated_info.get("tail_low_energy_seconds", 0.0))
        cap_hit = bool(row.get("cap_hit", False))
        inferred_anomaly = (
            cap_hit
            or duration_ratio > duration_ratio_anomaly_threshold
            or generated_info["seconds"] <= 0.0
            or float(generated_info.get("peak", 0.0)) >= peak_warn_threshold
            or float(generated_info.get("clipped_frac", 0.0)) > clipped_frac_warn_threshold
            or float(generated_info.get("hf_noise_ratio", 0.0)) > hf_noise_warn_threshold
            or float(generated_info.get("voiced_f0_delta_p95", 0.0)) > voiced_f0_delta_warn_threshold
        )
        report_rows.append(
            {
                "sample_id": sample_id,
                "text": row.get("text", ""),
                "target_audio": str(target_audio),
                "generated_audio": str(generated_path),
                "target_seconds": round(target_info["seconds"], 6),
                "decoded_seconds": round(generated_info["seconds"], 6),
                "generated_seconds": round(generated_info["seconds"], 6),
                "duration_ratio": round(duration_ratio, 6),
                "duration_abs_error": round(abs(generated_info["seconds"] - target_info["seconds"]), 6),
                "generated_tail_low_energy_seconds": round(trailing_seconds, 6),
                "generated_peak": float(generated_info.get("peak", 0.0)),
                "generated_clipped_frac": float(generated_info.get("clipped_frac", 0.0)),
                "generated_hf_noise_ratio": float(generated_info.get("hf_noise_ratio", 0.0)),
                "generated_voiced_f0_delta_p95": float(generated_info.get("voiced_f0_delta_p95", 0.0)),
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
        "max_generated_peak": max((row["generated_peak"] for row in report_rows), default=0.0),
        "max_generated_clipped_frac": max((row["generated_clipped_frac"] for row in report_rows), default=0.0),
        "max_generated_hf_noise_ratio": max((row["generated_hf_noise_ratio"] for row in report_rows), default=0.0),
        "max_generated_voiced_f0_delta_p95": max(
            (row["generated_voiced_f0_delta_p95"] for row in report_rows),
            default=0.0,
        ),
        "duration_ratio_anomaly_threshold": duration_ratio_anomaly_threshold,
        "peak_warn_threshold": peak_warn_threshold,
        "clipped_frac_warn_threshold": clipped_frac_warn_threshold,
        "hf_noise_warn_threshold": hf_noise_warn_threshold,
        "voiced_f0_delta_warn_threshold": voiced_f0_delta_warn_threshold,
        "samples": report_rows,
    }
    return summary, report_rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed_eval_dir", type=str, required=True, help="checkpoint eval dir, supports fixed_eval_audio or free_run_eval_audio")
    parser.add_argument("--fixed_eval_jsonl", type=str, default=None)
    parser.add_argument("--reference_jsonl", type=str, default=None)
    parser.add_argument("--raw_audio_root", type=str, default="assets/BZNSYP/Wave")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--duration_ratio_anomaly_threshold", type=float, default=1.5)
    parser.add_argument("--peak_warn_threshold", type=float, default=0.99)
    parser.add_argument("--clipped_frac_warn_threshold", type=float, default=1e-6)
    parser.add_argument("--hf_noise_warn_threshold", type=float, default=0.12)
    parser.add_argument("--voiced_f0_delta_warn_threshold", type=float, default=180.0)
    args = parser.parse_args(argv)

    fixed_eval_dir = Path(args.fixed_eval_dir)
    fixed_eval_jsonl = Path(args.fixed_eval_jsonl) if args.fixed_eval_jsonl else None
    reference_jsonl = Path(args.reference_jsonl) if args.reference_jsonl else None
    raw_audio_root = Path(args.raw_audio_root) if args.raw_audio_root else None

    summary, report_rows = diagnose_eval_dir(
        fixed_eval_dir=fixed_eval_dir,
        fixed_eval_jsonl=fixed_eval_jsonl,
        reference_jsonl=reference_jsonl,
        raw_audio_root=raw_audio_root,
        duration_ratio_anomaly_threshold=args.duration_ratio_anomaly_threshold,
        peak_warn_threshold=args.peak_warn_threshold,
        clipped_frac_warn_threshold=args.clipped_frac_warn_threshold,
        hf_noise_warn_threshold=args.hf_noise_warn_threshold,
        voiced_f0_delta_warn_threshold=args.voiced_f0_delta_warn_threshold,
    )

    output_json = Path(args.output_json) if args.output_json else fixed_eval_dir / "diagnosis_report.json"
    output_csv = Path(args.output_csv) if args.output_csv else fixed_eval_dir / "diagnosis_report.csv"
    write_json(output_json, summary)
    write_csv(output_csv, report_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
