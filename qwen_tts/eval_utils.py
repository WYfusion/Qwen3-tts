# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def build_custom_voice_decode_kwargs(*, max_new_tokens: int, do_sample: bool = False) -> dict[str, Any]:
    return {
        "do_sample": bool(do_sample),
        "subtalker_dosample": bool(do_sample),
        "max_new_tokens": int(max_new_tokens),
    }


def compute_length_cap(
    row: dict[str, Any],
    *,
    length_mode: str = "dynamic",
    length_multiplier: float = 2.0,
    fixed_max_new_tokens: int = 256,
) -> int:
    if length_mode == "dynamic" and "target_code_frames" in row:
        dynamic_cap = math.ceil(float(row["target_code_frames"]) * float(length_multiplier))
        return int(max(2, min(fixed_max_new_tokens, dynamic_cap)))
    return int(fixed_max_new_tokens)


def _to_mono_float32(wav: np.ndarray) -> np.ndarray:
    arr = np.asarray(wav, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr


def trailing_low_energy_seconds_from_array(
    wav: np.ndarray,
    sample_rate: int,
    *,
    frame_seconds: float = 0.05,
    threshold_db: float = -35.0,
) -> float:
    arr = _to_mono_float32(wav)
    if arr.size == 0 or sample_rate <= 0:
        return 0.0
    hop = max(1, int(sample_rate * frame_seconds))
    frame_rms = []
    for start in range(0, len(arr), hop):
        chunk = arr[start : start + hop]
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


def hf_noise_ratio_from_array(
    wav: np.ndarray,
    sample_rate: int,
    *,
    cutoff_hz: float = 6000.0,
) -> float:
    arr = _to_mono_float32(wav)
    if arr.size == 0 or sample_rate <= 0:
        return 0.0
    spectrum = np.fft.rfft(arr)
    power = np.abs(spectrum) ** 2
    total_power = float(np.sum(power))
    if total_power <= 1e-12:
        return 0.0
    freqs = np.fft.rfftfreq(arr.shape[0], d=1.0 / float(sample_rate))
    hf_power = float(np.sum(power[freqs >= cutoff_hz]))
    return round(hf_power / total_power, 6)


def voiced_f0_delta_p95_from_array(
    wav: np.ndarray,
    sample_rate: int,
    *,
    fmin: float = 60.0,
    fmax: float = 500.0,
) -> float:
    arr = _to_mono_float32(wav)
    if arr.size < max(32, sample_rate // 8) or sample_rate <= 0:
        return 0.0
    try:
        import librosa
    except ModuleNotFoundError:
        return 0.0

    frame_length = min(2048, int(2 ** np.floor(np.log2(max(256, arr.size)))))
    hop_length = max(128, sample_rate // 100)
    try:
        f0 = librosa.yin(
            arr,
            fmin=fmin,
            fmax=fmax,
            sr=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    except Exception:
        return 0.0
    f0 = np.asarray(f0, dtype=np.float32)
    voiced = f0[np.isfinite(f0) & (f0 > 0)]
    if voiced.size < 3:
        return 0.0
    cents = np.abs(np.diff(np.log2(voiced))) * 1200.0
    if cents.size == 0:
        return 0.0
    return round(float(np.percentile(cents, 95.0)), 6)


def summarize_audio_array(wav: np.ndarray, sample_rate: int, *, clip_threshold: float = 0.999) -> dict[str, float | int]:
    arr = _to_mono_float32(wav)
    frames = int(arr.shape[0])
    if frames == 0:
        peak = 0.0
        clipped_frac = 0.0
        rms = 0.0
    else:
        peak = float(np.max(np.abs(arr)))
        clipped_frac = float(np.mean(np.abs(arr) >= clip_threshold))
        rms = float(np.sqrt(np.mean(np.square(arr))) + 1e-12)
    return {
        "frames": frames,
        "sample_rate": int(sample_rate),
        "decoded_seconds": round(frames / float(sample_rate), 6) if sample_rate else 0.0,
        "peak": round(peak, 6),
        "clipped_frac": round(clipped_frac, 6),
        "rms": round(rms, 6),
        "tail_low_energy_seconds": trailing_low_energy_seconds_from_array(arr, sample_rate),
        "hf_noise_ratio": hf_noise_ratio_from_array(arr, sample_rate),
        "voiced_f0_delta_p95": voiced_f0_delta_p95_from_array(arr, sample_rate),
    }


def summarize_audio_path(path: str | Path, *, clip_threshold: float = 0.999) -> dict[str, float | int]:
    audio_path = Path(path)
    info = sf.info(str(audio_path))
    wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    metrics = summarize_audio_array(wav, sr, clip_threshold=clip_threshold)
    metrics["frames"] = int(info.frames)
    metrics["sample_rate"] = int(info.samplerate)
    metrics["decoded_seconds"] = round(float(info.frames) / float(info.samplerate), 6)
    return metrics
