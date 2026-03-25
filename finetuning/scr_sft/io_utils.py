# coding=utf-8

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f]


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")


def json_ready(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if torch.is_tensor(obj):
        tensor = obj.detach().cpu()
        if tensor.numel() <= 32:
            return tensor.tolist()
        tensor_f = tensor.float()
        return {
            "_type": "tensor",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "min": float(tensor_f.min().item()),
            "max": float(tensor_f.max().item()),
            "mean": float(tensor_f.mean().item()),
            "std": float(tensor_f.std().item()) if tensor.numel() > 1 else 0.0,
        }
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    return str(obj)


def append_metrics_csv(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        fieldnames = next(reader, [])
    new_keys = [key for key in row.keys() if key not in fieldnames]
    if not new_keys:
        with csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
        return

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)
    fieldnames.extend(new_keys)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for existing in existing_rows:
            writer.writerow(existing)
        writer.writerow(row)


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_round(value, digits=8):
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return round(value, digits)


def detach_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: detach_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [detach_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(detach_to_cpu(v) for v in obj)
    return obj


def resolve_media_path(path_value: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    if base_dir is not None:
        candidate = Path(base_dir) / path
        if candidate.exists():
            return candidate.resolve()
        return candidate
    return path


def audio_info(path: str | Path):
    import soundfile as sf

    info = sf.info(str(path))
    seconds = float(info.frames) / float(info.samplerate)
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "seconds": seconds,
        "frames": int(info.frames),
    }

