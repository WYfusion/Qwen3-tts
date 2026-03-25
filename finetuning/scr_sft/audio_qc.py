# coding=utf-8

from __future__ import annotations

from pathlib import Path

from qwen_tts.eval_utils import summarize_audio_path

from .io_utils import resolve_media_path, safe_round, write_csv, write_json


def build_audio_qc_report(train_data, *, output_dir: Path, peak_warn: float, clip_warn: float):
    rows = []
    for idx, item in enumerate(train_data):
        audio_path = resolve_media_path(item["audio"], base_dir=Path.cwd())
        metrics = summarize_audio_path(audio_path)
        ref_audio = item.get("ref_audio", "")
        ref_audio_path = resolve_media_path(ref_audio, base_dir=Path.cwd()) if ref_audio else ""
        rows.append(
            {
                "index": idx,
                "audio": str(audio_path),
                "ref_audio": "" if ref_audio_path == "" else str(ref_audio_path),
                "text_chars": len(item.get("text", "")),
                "code_frames": len(item.get("audio_codes", [])),
                "decoded_seconds": safe_round(metrics.get("decoded_seconds", 0.0), 6),
                "peak": safe_round(metrics.get("peak", 0.0), 6),
                "clipped_frac": safe_round(metrics.get("clipped_frac", 0.0), 8),
                "rms": safe_round(metrics.get("rms", 0.0), 6),
                "tail_low_energy_seconds": safe_round(metrics.get("tail_low_energy_seconds", 0.0), 6),
                "hf_noise_ratio": safe_round(metrics.get("hf_noise_ratio", 0.0), 6),
                "voiced_f0_delta_p95": safe_round(metrics.get("voiced_f0_delta_p95", 0.0), 6),
                "peak_warn": bool(float(metrics.get("peak", 0.0)) >= float(peak_warn)),
                "clip_warn": bool(float(metrics.get("clipped_frac", 0.0)) > float(clip_warn)),
            }
        )

    rows.sort(
        key=lambda item: (
            bool(item["clip_warn"]),
            float(item["peak"]),
            float(item["hf_noise_ratio"]),
            float(item["voiced_f0_delta_p95"]),
        ),
        reverse=True,
    )

    def _mean(key: str) -> float:
        values = [float(row[key]) for row in rows]
        return float(sum(values) / max(1, len(values)))

    def _max(key: str) -> float:
        values = [float(row[key]) for row in rows]
        return float(max(values) if values else 0.0)

    summary = {
        "num_samples": len(rows),
        "mean_decoded_seconds": safe_round(_mean("decoded_seconds"), 6),
        "max_decoded_seconds": safe_round(_max("decoded_seconds"), 6),
        "mean_peak": safe_round(_mean("peak"), 6),
        "max_peak": safe_round(_max("peak"), 6),
        "mean_clipped_frac": safe_round(_mean("clipped_frac"), 8),
        "max_clipped_frac": safe_round(_max("clipped_frac"), 8),
        "mean_rms": safe_round(_mean("rms"), 6),
        "mean_tail_low_energy_seconds": safe_round(_mean("tail_low_energy_seconds"), 6),
        "max_tail_low_energy_seconds": safe_round(_max("tail_low_energy_seconds"), 6),
        "mean_hf_noise_ratio": safe_round(_mean("hf_noise_ratio"), 6),
        "max_hf_noise_ratio": safe_round(_max("hf_noise_ratio"), 6),
        "mean_voiced_f0_delta_p95": safe_round(_mean("voiced_f0_delta_p95"), 6),
        "max_voiced_f0_delta_p95": safe_round(_max("voiced_f0_delta_p95"), 6),
        "num_peak_warn": sum(1 for row in rows if row["peak_warn"]),
        "num_clip_warn": sum(1 for row in rows if row["clip_warn"]),
        "peak_warn_threshold": peak_warn,
        "clipped_frac_warn_threshold": clip_warn,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "samples": rows}
    write_json(output_dir / "audio_qc_report.json", payload)
    write_csv(output_dir / "audio_qc_report.csv", rows)
    return payload
