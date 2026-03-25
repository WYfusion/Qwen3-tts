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


def summarize_audio_array(wav: np.ndarray, sample_rate: int, *, clip_threshold: float = 0.999) -> dict[str, float | int]:
    arr = _to_mono_float32(wav)
    frames = int(arr.shape[0])
    if frames == 0:
        peak = 0.0
        clipped_frac = 0.0
    else:
        peak = float(np.max(np.abs(arr)))
        clipped_frac = float(np.mean(np.abs(arr) >= clip_threshold))
    return {
        "frames": frames,
        "sample_rate": int(sample_rate),
        "decoded_seconds": round(frames / float(sample_rate), 6) if sample_rate else 0.0,
        "peak": round(peak, 6),
        "clipped_frac": round(clipped_frac, 6),
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
