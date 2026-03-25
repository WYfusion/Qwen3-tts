# coding=utf-8

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
from qwen_tts import Qwen3TTSModel
from qwen_tts.eval_utils import summarize_audio_array

from .constants import CUSTOM_SPEAKER_ID
from .io_utils import safe_round, write_json
from .recipes import effective_num_epochs


def fixed_eval_cap(row: dict, eval_config) -> int:
    if eval_config.fixed_eval_length_mode == "dynamic":
        dynamic_cap = round(float(row["target_code_frames"]) * float(eval_config.fixed_eval_length_multiplier))
        return int(max(2, min(int(eval_config.fixed_eval_max_new_tokens), int(dynamic_cap))))
    return int(eval_config.fixed_eval_max_new_tokens)


def compute_qc_score(summary: dict, eval_config) -> float:
    fail_rate = float(summary["num_failed_samples"]) / max(1, int(summary["num_samples"]))
    noise_penalty = max(0.0, float(summary["mean_hf_noise_ratio"]) - float(eval_config.hf_noise_warn_threshold))
    jitter_penalty = max(
        0.0,
        float(summary["mean_voiced_f0_delta_p95"]) - float(eval_config.voiced_f0_delta_warn_threshold),
    ) / max(float(eval_config.voiced_f0_delta_warn_threshold), 1.0)
    peak_penalty = 1.0 if float(summary["max_peak"]) >= float(eval_config.peak_warn_threshold) else 0.0
    clip_penalty = 1.0 if float(summary["max_clipped_frac"]) > float(eval_config.clipped_frac_warn_threshold) else 0.0
    return (
        float(summary["max_duration_ratio"])
        + 5.0 * float(summary["cap_hit_rate"])
        + 2.0 * fail_rate
        + noise_penalty
        + jitter_penalty
        + peak_penalty
        + clip_penalty
    )


def build_generation_kwargs_for_eval(eval_name: str, row: dict, eval_config):
    if eval_name == "fixed_eval":
        sample_cap = fixed_eval_cap(row, eval_config)
        kwargs = {
            "do_sample": bool(eval_config.fixed_eval_do_sample),
            "subtalker_dosample": bool(eval_config.fixed_eval_do_sample),
            "max_new_tokens": int(sample_cap),
        }
        if not eval_config.fixed_eval_do_sample:
            kwargs.update(
                {
                    "temperature": 1.0,
                    "subtalker_temperature": 1.0,
                    "top_k": 1,
                    "subtalker_top_k": 1,
                    "top_p": 1.0,
                    "subtalker_top_p": 1.0,
                }
            )
        return int(sample_cap), kwargs
    if eval_name == "free_run_eval":
        cap = int(eval_config.free_run_eval_max_new_tokens)
        return cap, {"max_new_tokens": cap}
    raise ValueError(f"Unsupported eval_name: {eval_name}")


def generate_eval_audio(tts, row: dict, speaker_name: str, generation_kwargs: dict[str, Any]):
    text_ids = tts._tokenize_texts([tts._build_assistant_text(row["text"])])
    instruct = row.get("instruct", "")
    instruct_ids = [None]
    if instruct:
        instruct_ids = [tts._tokenize_texts([tts._build_instruct_text(instruct)])[0]]

    gen_kwargs = tts._merge_generate_kwargs(**generation_kwargs)
    talker_codes_list, _ = tts.model.generate(
        input_ids=text_ids,
        instruct_ids=instruct_ids,
        languages=[row.get("language", "Chinese")],
        speakers=[row.get("speaker", speaker_name)],
        non_streaming_mode=True,
        **gen_kwargs,
    )
    talker_codes = talker_codes_list[0]
    wavs, sample_rate = tts.model.speech_tokenizer.decode([{"audio_codes": talker_codes}])
    return wavs[0], sample_rate, talker_codes


def summarize_eval_manifest(manifest_rows, eval_config):
    if not manifest_rows:
        return {
            "num_samples": 0,
            "num_failed_samples": 0,
            "cap_hit_rate": 0.0,
            "mean_duration_ratio": 0.0,
            "max_duration_ratio": 0.0,
            "mean_abs_duration_error": 0.0,
            "mean_decoded_seconds": 0.0,
            "max_peak": 0.0,
            "mean_peak": 0.0,
            "max_clipped_frac": 0.0,
            "mean_clipped_frac": 0.0,
            "mean_hf_noise_ratio": 0.0,
            "max_hf_noise_ratio": 0.0,
            "mean_voiced_f0_delta_p95": 0.0,
            "max_voiced_f0_delta_p95": 0.0,
            "qc_score": 0.0,
        }

    def _collect(key: str):
        return [float(row[key]) for row in manifest_rows if row.get(key) not in ("", None)]

    ratios = _collect("duration_ratio")
    abs_errors = _collect("duration_abs_error")
    decoded_seconds = _collect("decoded_seconds")
    peaks = _collect("peak")
    clipped = _collect("clipped_frac")
    hf_noise = _collect("hf_noise_ratio")
    voiced = _collect("voiced_f0_delta_p95")
    cap_hits = sum(1 for row in manifest_rows if bool(row.get("cap_hit")))
    failed = sum(1 for row in manifest_rows if bool(row.get("is_anomaly")))
    summary = {
        "num_samples": len(manifest_rows),
        "num_failed_samples": failed,
        "cap_hit_rate": float(cap_hits) / max(1, len(manifest_rows)),
        "mean_duration_ratio": sum(ratios) / max(1, len(ratios)),
        "max_duration_ratio": max(ratios) if ratios else 0.0,
        "mean_abs_duration_error": sum(abs_errors) / max(1, len(abs_errors)),
        "mean_decoded_seconds": sum(decoded_seconds) / max(1, len(decoded_seconds)),
        "max_peak": max(peaks) if peaks else 0.0,
        "mean_peak": sum(peaks) / max(1, len(peaks)),
        "max_clipped_frac": max(clipped) if clipped else 0.0,
        "mean_clipped_frac": sum(clipped) / max(1, len(clipped)),
        "mean_hf_noise_ratio": sum(hf_noise) / max(1, len(hf_noise)),
        "max_hf_noise_ratio": max(hf_noise) if hf_noise else 0.0,
        "mean_voiced_f0_delta_p95": sum(voiced) / max(1, len(voiced)),
        "max_voiced_f0_delta_p95": max(voiced) if voiced else 0.0,
    }
    summary["qc_score"] = compute_qc_score(summary, eval_config)
    return summary


def log_eval_to_wandb(
    accelerator,
    *,
    eval_name: str,
    manifest_rows,
    global_step: int,
    epoch: int,
    log_audio: bool,
):
    tracker = None
    try:
        tracker = accelerator.get_tracker("wandb")
    except Exception:
        tracker = None
    if tracker is None:
        return

    import wandb

    columns = [
        "epoch",
        "sample_id",
        "text",
        "target_seconds",
        "decoded_seconds",
        "duration_ratio",
        "duration_abs_error",
        "generated_code_frames",
        "cap_hit",
        "peak",
        "clipped_frac",
        "hf_noise_ratio",
        "voiced_f0_delta_p95",
        "is_anomaly",
        "target_audio",
        "generated_audio",
    ]
    data = []
    for row in manifest_rows:
        target_audio = row.get("audio", "")
        generated_audio = row.get("wav_path", "")
        if log_audio:
            target_audio = wandb.Audio(str(target_audio), sample_rate=int(row.get("target_sample_rate", 24000)))
            generated_audio = wandb.Audio(
                str(generated_audio),
                sample_rate=int(row.get("generated_sample_rate", row.get("target_sample_rate", 24000))),
            )
        data.append(
            [
                epoch,
                row.get("sample_id", ""),
                row.get("text", ""),
                row.get("target_seconds", ""),
                row.get("decoded_seconds", ""),
                row.get("duration_ratio", ""),
                row.get("duration_abs_error", ""),
                row.get("generated_code_frames", ""),
                bool(row.get("cap_hit")),
                row.get("peak", ""),
                row.get("clipped_frac", ""),
                row.get("hf_noise_ratio", ""),
                row.get("voiced_f0_delta_p95", ""),
                bool(row.get("is_anomaly")),
                target_audio,
                generated_audio,
            ]
        )
    tracker.log_table(
        table_name=f"{eval_name}/samples_epoch_{epoch}",
        columns=columns,
        data=data,
        step=global_step,
    )


def render_eval_samples(
    *,
    accelerator,
    model,
    processor,
    speaker_name: str,
    eval_rows,
    output_dir: Path,
    eval_config,
    train_config,
    global_step: int,
    epoch: int,
    eval_name: str,
):
    if not accelerator.is_main_process or not eval_rows:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    previous_mode = unwrapped_model.training
    previous_tts_model_type = unwrapped_model.tts_model_type
    previous_config_tts_model_type = unwrapped_model.config.tts_model_type
    previous_spk_id = dict(unwrapped_model.config.talker_config.spk_id)
    previous_spk_is_dialect = dict(unwrapped_model.config.talker_config.spk_is_dialect)
    previous_supported_speakers = unwrapped_model.supported_speakers

    with torch.no_grad():
        unwrapped_model.config.tts_model_type = "custom_voice"
        unwrapped_model.tts_model_type = "custom_voice"
        unwrapped_model.config.talker_config.spk_id = {speaker_name: CUSTOM_SPEAKER_ID}
        unwrapped_model.config.talker_config.spk_is_dialect = {speaker_name: False}
        unwrapped_model.supported_speakers = unwrapped_model.config.talker_config.spk_id.keys()

    try:
        import soundfile as sf

        unwrapped_model.eval()
        tts = Qwen3TTSModel(unwrapped_model, processor, generate_defaults=unwrapped_model.generate_config)
        manifest = []
        for idx, row in enumerate(eval_rows):
            sample_cap, generation_kwargs = build_generation_kwargs_for_eval(eval_name, row, eval_config)
            accelerator.print(
                f"[{eval_name}] sample {idx + 1}/{len(eval_rows)} | sample_id={row['sample_id']} | "
                f"max_new_tokens={sample_cap}"
            )
            sample_start = time.perf_counter()
            wav, sample_rate, talker_codes = generate_eval_audio(
                tts=tts,
                row=row,
                speaker_name=speaker_name,
                generation_kwargs=generation_kwargs,
            )
            wav_path = output_dir / f"{row['sample_id']}.wav"
            sf.write(wav_path, wav, sample_rate)
            audio_metrics = summarize_audio_array(wav, sample_rate)
            decoded_seconds = float(audio_metrics["decoded_seconds"])
            target_seconds = float(row["target_seconds"])
            duration_ratio = decoded_seconds / max(target_seconds, 1e-6)
            duration_abs_error = abs(decoded_seconds - target_seconds)
            generated_code_frames = int(talker_codes.shape[0]) if hasattr(talker_codes, "shape") else int(len(talker_codes))
            cap_hit = generated_code_frames >= int(sample_cap)
            stop_reason = "cap_reached" if cap_hit else "eos_or_stop"
            is_anomaly = (
                cap_hit
                or decoded_seconds <= 0.0
                or duration_ratio > float(eval_config.fixed_eval_duration_ratio_warn)
                or float(audio_metrics["peak"]) >= float(eval_config.peak_warn_threshold)
                or float(audio_metrics["clipped_frac"]) > float(eval_config.clipped_frac_warn_threshold)
                or float(audio_metrics["hf_noise_ratio"]) > float(eval_config.hf_noise_warn_threshold)
                or float(audio_metrics["voiced_f0_delta_p95"]) > float(eval_config.voiced_f0_delta_warn_threshold)
            )
            manifest.append(
                {
                    "index": idx,
                    "sample_id": row["sample_id"],
                    "wav_path": str(wav_path),
                    "text": row["text"],
                    "language": row.get("language", eval_config.fixed_eval_language),
                    "instruct": row.get("instruct", ""),
                    "speaker": row.get("speaker", speaker_name),
                    "audio": row["audio"],
                    "ref_audio": row["ref_audio"],
                    "target_seconds": safe_round(target_seconds, 6),
                    "decoded_seconds": safe_round(decoded_seconds, 6),
                    "generated_seconds": safe_round(decoded_seconds, 6),
                    "duration_ratio": safe_round(duration_ratio, 6),
                    "duration_abs_error": safe_round(duration_abs_error, 6),
                    "target_code_frames": int(row["target_code_frames"]),
                    "generated_code_frames": generated_code_frames,
                    "target_sample_rate": row.get("target_sample_rate", 24000),
                    "generated_sample_rate": sample_rate,
                    "max_new_tokens": int(sample_cap),
                    "cap_hit": cap_hit,
                    "stop_reason": stop_reason,
                    "is_anomaly": is_anomaly,
                    "generation_wall_time": round(time.perf_counter() - sample_start, 4),
                    "peak": audio_metrics["peak"],
                    "clipped_frac": audio_metrics["clipped_frac"],
                    "rms": audio_metrics.get("rms", 0.0),
                    "tail_low_energy_seconds": audio_metrics.get("tail_low_energy_seconds", 0.0),
                    "hf_noise_ratio": audio_metrics.get("hf_noise_ratio", 0.0),
                    "voiced_f0_delta_p95": audio_metrics.get("voiced_f0_delta_p95", 0.0),
                }
            )
        summary = summarize_eval_manifest(manifest, eval_config)
        write_json(output_dir / "manifest.json", {"summary": summary, "samples": manifest})
        is_sweep_run = bool(os.environ.get("WANDB_SWEEP_ID"))
        log_audio = not is_sweep_run or epoch == effective_num_epochs(train_config) - 1
        log_eval_to_wandb(
            accelerator,
            eval_name=eval_name,
            manifest_rows=manifest,
            global_step=global_step,
            epoch=epoch,
            log_audio=log_audio,
        )
        return {"summary": summary, "samples": manifest}
    finally:
        with torch.no_grad():
            unwrapped_model.config.tts_model_type = previous_config_tts_model_type
            unwrapped_model.tts_model_type = previous_tts_model_type
            unwrapped_model.config.talker_config.spk_id = previous_spk_id
            unwrapped_model.config.talker_config.spk_is_dialect = previous_spk_is_dialect
            unwrapped_model.supported_speakers = previous_supported_speakers
        if previous_mode:
            unwrapped_model.train()
