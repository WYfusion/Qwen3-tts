# coding=utf-8

from __future__ import annotations

import math
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import infer_benchmark_eval_source_path, row_duration_seconds
from .io_utils import read_json, read_jsonl, write_json, write_jsonl

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class BenchmarkMetrics:
    checkpoint_dir: str
    eval_jsonl: str
    output_dir: str
    label: str
    wer: float
    asv_mean: float
    asv_std: float
    num_samples: int
    wer_summary_path: str
    sim_summary_path: str
    manifest_path: str


def _resolve_path(path_value: str | Path, *, base_dir: Path | None = None, must_exist: bool = True) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        resolved = path.resolve()
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Path not found: {resolved}")
        return resolved

    candidates = [Path.cwd()]
    if base_dir is not None:
        candidates.append(base_dir)
    candidates.append(REPO_ROOT)
    for root in candidates:
        candidate = (root / path).resolve()
        if not must_exist or candidate.exists():
            return candidate
    raise FileNotFoundError(f"Path not found: {path}")


def _resolve_optional_existing_path(path_value: str | Path, *, candidate_roots: list[Path]) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    direct = path.resolve()
    if direct.exists():
        return direct
    for root in candidate_roots:
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    return (candidate_roots[0] / path).resolve() if candidate_roots else direct


def _utt_from_row(row: dict) -> str:
    for key in ("utt", "sample_id", "audio"):
        value = row.get(key)
        if value:
            return Path(str(value)).stem
    raise KeyError(f"Could not infer utt from row keys={sorted(row.keys())}")


def parse_wer_raw(path: str | Path) -> dict[str, float]:
    rows = {}
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if parts[0] == "utt":
                continue
            if len(parts) < 2:
                continue
            utt = Path(parts[0]).stem
            rows[utt] = float(parts[1]) * 100.0
    return rows


def parse_wer_summary(path: str | Path) -> float:
    wer_by_utt = parse_wer_raw(path)
    if not wer_by_utt:
        raise ValueError(f"No per-utt WER rows found in {path}")
    return sum(wer_by_utt.values()) / float(len(wer_by_utt))


def parse_sim_summary(path: str | Path) -> tuple[float, float]:
    scores = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                scores.append(float(parts[1]))
            except ValueError:
                continue
    if not scores:
        raise ValueError(f"No SIM scores found in {path}")
    mean = float(sum(scores) / len(scores))
    std = float(statistics.pstdev(scores)) if len(scores) > 1 else 0.0
    return mean, std


def _relative_gain(base_value: float, current_value: float) -> float:
    if abs(float(base_value)) < 1e-12:
        return 0.0
    return (float(current_value) - float(base_value)) / float(base_value)


def build_benchmark_selection_record(base_metrics: BenchmarkMetrics, checkpoint_metrics: BenchmarkMetrics, free_run_summary: dict | None):
    free_run_summary = free_run_summary or {}
    base_wer = float(base_metrics.wer)
    base_asv = float(base_metrics.asv_mean)
    benchmark_wer = float(checkpoint_metrics.wer)
    benchmark_asv = float(checkpoint_metrics.asv_mean)
    wer_gain = (base_wer - benchmark_wer) / base_wer if abs(base_wer) > 1e-12 else 0.0
    asv_gain = _relative_gain(base_asv, benchmark_asv)
    objective = 0.6 * wer_gain + 0.4 * asv_gain
    cap_hit_rate = float(free_run_summary.get("cap_hit_rate", 0.0))
    max_duration_ratio = float(free_run_summary.get("max_duration_ratio", 0.0))
    is_eligible = cap_hit_rate == 0.0 and max_duration_ratio <= 1.8
    beats_base_both = benchmark_wer < base_wer and benchmark_asv > base_asv
    return {
        "benchmark_success": True,
        "benchmark_base_wer": base_wer,
        "benchmark_base_asv": base_asv,
        "benchmark_wer": benchmark_wer,
        "benchmark_asv": benchmark_asv,
        "benchmark_asv_std": float(checkpoint_metrics.asv_std),
        "benchmark_num_samples": int(checkpoint_metrics.num_samples),
        "benchmark_output_dir": str(checkpoint_metrics.output_dir),
        "benchmark_manifest_path": str(checkpoint_metrics.manifest_path),
        "wer_gain": wer_gain,
        "asv_gain": asv_gain,
        "benchmark_objective": objective,
        "free_run_qc_score": float(free_run_summary.get("qc_score", math.inf)),
        "is_benchmark_eligible": bool(is_eligible),
        "beats_base_both": bool(beats_base_both),
    }


def should_replace_benchmark_best(candidate: dict | None, incumbent: dict | None) -> bool:
    if not candidate or not candidate.get("is_benchmark_eligible") or not candidate.get("beats_base_both"):
        return False
    if not incumbent or not incumbent.get("is_benchmark_eligible") or not incumbent.get("beats_base_both"):
        return True

    cand_wer = float(candidate.get("benchmark_wer", math.inf))
    inc_wer = float(incumbent.get("benchmark_wer", math.inf))
    if cand_wer < inc_wer - 1e-12:
        return True
    if cand_wer > inc_wer + 1e-12:
        return False

    cand_asv = float(candidate.get("benchmark_asv", -math.inf))
    inc_asv = float(incumbent.get("benchmark_asv", -math.inf))
    if cand_asv > inc_asv + 1e-12:
        return True
    if cand_asv < inc_asv - 1e-12:
        return False

    cand_qc = float(candidate.get("free_run_qc_score", math.inf))
    inc_qc = float(incumbent.get("free_run_qc_score", math.inf))
    if cand_qc < inc_qc - 1e-12:
        return True
    if cand_qc > inc_qc + 1e-12:
        return False

    return float(candidate.get("benchmark_objective", -math.inf)) > float(
        incumbent.get("benchmark_objective", -math.inf)
    )


def _split_sorted_rows(rows: list[dict], *, groups: int) -> list[list[dict]]:
    if groups <= 0:
        return []
    ordered = sorted(
        rows,
        key=lambda row: (-float(row.get("base_wer", 0.0)), row_duration_seconds(row), _utt_from_row(row)),
    )
    total = len(ordered)
    base = total // groups
    remainder = total % groups
    buckets = []
    start = 0
    for bucket_idx in range(groups):
        size = base + (1 if bucket_idx < remainder else 0)
        end = start + size
        buckets.append(ordered[start:end])
        start = end
    return buckets


def _allocate_counts(total: int, bucket_sizes: list[int]) -> list[int]:
    if total <= 0 or not bucket_sizes:
        return [0 for _ in bucket_sizes]
    total_available = sum(bucket_sizes)
    if total_available <= total:
        return list(bucket_sizes)
    base = total // len(bucket_sizes)
    remainder = total % len(bucket_sizes)
    counts = []
    for idx, size in enumerate(bucket_sizes):
        counts.append(min(size, base + (1 if idx < remainder else 0)))
    remaining = total - sum(counts)
    while remaining > 0:
        progressed = False
        for idx, size in enumerate(bucket_sizes):
            if counts[idx] >= size:
                continue
            counts[idx] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return counts


def _pick_evenly_spaced(rows: list[dict], count: int) -> list[dict]:
    if count <= 0 or not rows:
        return []
    if count >= len(rows):
        return list(rows)
    if count == 1:
        return [rows[len(rows) // 2]]
    selected = []
    used = set()
    for idx in range(count):
        pos = round(idx * (len(rows) - 1) / (count - 1))
        pos = max(0, min(len(rows) - 1, int(pos)))
        while pos in used and pos + 1 < len(rows):
            pos += 1
        if pos in used:
            pos = min(j for j in range(len(rows)) if j not in used)
        used.add(pos)
        selected.append(rows[pos])
    return selected


def derive_benchmark_subset(*, eval_rows: list[dict], base_manifest: dict, base_wer_by_utt: dict[str, float], sample_count: int):
    samples_by_utt = {
        str(sample.get("utt")): sample for sample in base_manifest.get("samples", []) if sample.get("utt")
    }
    annotated = []
    for row in eval_rows:
        utt = _utt_from_row(row)
        sample_meta = samples_by_utt.get(utt, {})
        annotated.append(
            {
                **row,
                "utt": utt,
                "base_wer": float(base_wer_by_utt.get(utt, 0.0)),
                "target_seconds": float(sample_meta.get("target_seconds", row_duration_seconds(row))),
            }
        )

    slices = _split_sorted_rows(annotated, groups=min(3, len(annotated)))
    slice_names = ["hard", "medium", "easy"]
    slice_counts = _allocate_counts(sample_count, [len(bucket) for bucket in slices])
    subset_rows = []
    manifest_rows = []
    for slice_idx, (slice_rows, slice_count) in enumerate(zip(slices, slice_counts)):
        if slice_count <= 0 or not slice_rows:
            continue
        by_duration = sorted(slice_rows, key=lambda row: (row_duration_seconds(row), row["utt"]))
        duration_buckets = []
        total = len(by_duration)
        base = total // 3
        remainder = total % 3
        start = 0
        for bucket_idx in range(3):
            size = base + (1 if bucket_idx < remainder else 0)
            end = start + size
            duration_buckets.append(by_duration[start:end])
            start = end
        duration_counts = _allocate_counts(slice_count, [len(bucket) for bucket in duration_buckets])
        for bucket_idx, (bucket_rows, bucket_count) in enumerate(zip(duration_buckets, duration_counts)):
            for row in _pick_evenly_spaced(bucket_rows, bucket_count):
                subset_row = dict(row)
                subset_row.pop("base_wer", None)
                subset_rows.append(subset_row)
                manifest_rows.append(
                    {
                        "utt": row["utt"],
                        "base_wer": float(row["base_wer"]),
                        "target_seconds": float(row["target_seconds"]),
                        "source_slice": slice_names[min(slice_idx, len(slice_names) - 1)],
                        "duration_bucket": ["short", "mid", "long"][bucket_idx],
                        "text": row.get("text", ""),
                        "audio": row.get("audio", ""),
                    }
                )

    subset_rows = sorted(subset_rows, key=lambda row: _utt_from_row(row))
    manifest_rows = sorted(manifest_rows, key=lambda row: row["utt"])
    return subset_rows[:sample_count], manifest_rows[:sample_count]


class BenchmarkEvalManager:
    def __init__(self, *, train_config, eval_config, paths):
        self.train_config = train_config
        self.eval_config = eval_config
        self.paths = paths
        self.seed_tts_eval_root = _resolve_path(eval_config.seed_tts_eval_root, base_dir=REPO_ROOT)
        self.seed_tts_eval_python = str(eval_config.seed_tts_eval_python)
        self.sim_finetune_checkpoint = _resolve_optional_existing_path(
            eval_config.sim_finetune_checkpoint,
            candidate_roots=[
                Path.cwd(),
                self.seed_tts_eval_root,
                self.seed_tts_eval_root.parent,
                REPO_ROOT,
            ],
        )
        self._resolved_benchmark_eval_jsonl = None
        self._base_metrics = None
        self._subset_rows = None
        self._subset_path = None
        self._subset_manifest = None
        self._assets_prepared = False

    @property
    def enabled(self) -> bool:
        return int(self.eval_config.benchmark_eval_num_samples) > 0

    def _run(self, cmd: list[str], *, cwd: Path | None = None) -> str:
        result = subprocess.run(
            cmd,
            cwd=str(cwd or REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        return result.stdout

    def resolve_benchmark_eval_jsonl(self) -> Path:
        if self._resolved_benchmark_eval_jsonl is not None:
            return self._resolved_benchmark_eval_jsonl
        if self.eval_config.benchmark_eval_jsonl:
            path = _resolve_path(self.eval_config.benchmark_eval_jsonl, base_dir=REPO_ROOT)
        else:
            path = infer_benchmark_eval_source_path(self.train_config.train_jsonl).resolve()
        self._resolved_benchmark_eval_jsonl = path
        return path

    def verify_toolchain(self) -> None:
        if not self.enabled:
            return
        required = [
            REPO_ROOT / "finetuning" / "prepare_seed_tts_eval.py",
            self.seed_tts_eval_root / "get_wav_res_ref_text.py",
            self.seed_tts_eval_root / "run_wer.py",
            self.seed_tts_eval_root / "average_wer.py",
            self.seed_tts_eval_root / "prepare_ckpt.py",
            self.seed_tts_eval_root / "thirdparty" / "UniSpeech" / "downstreams" / "speaker_verification" / "verification_pair_list_v2.py",
            self.seed_tts_eval_root / "thirdparty" / "UniSpeech" / "downstreams" / "speaker_verification" / "average.py",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Benchmark eval toolchain is incomplete:\n" + "\n".join(missing))
        if not self.sim_finetune_checkpoint.exists():
            raise FileNotFoundError(
                "SIM finetune checkpoint not found: "
                f"{self.sim_finetune_checkpoint}\n"
                "Expected something like ./seed-tts-eval/weight/wavlm_large_finetune.pth "
                "or pass --sim_finetune_checkpoint explicitly."
            )
        self._run([self.seed_tts_eval_python, "--version"])

    def ensure_assets_prepared(self) -> None:
        if self._assets_prepared:
            return
        self._run(
            [
                self.seed_tts_eval_python,
                str(self.seed_tts_eval_root / "prepare_ckpt.py"),
                "--device",
                str(self.eval_config.benchmark_eval_device),
                "--prepare_sim",
                "--sim_model_name",
                "wavlm_large",
                "--sim_finetune_checkpoint",
                str(self.sim_finetune_checkpoint),
            ],
            cwd=REPO_ROOT,
        )
        self._assets_prepared = True

    def _build_metrics_from_dir(self, *, checkpoint_dir: Path, eval_jsonl: Path, output_dir: Path, label: str) -> BenchmarkMetrics:
        metrics_path = output_dir / "benchmark_metrics.json"
        if metrics_path.exists():
            return BenchmarkMetrics(**read_json(metrics_path))

        manifest_path = output_dir / "manifest.json"
        wer_summary_path = output_dir / "wer.summary.txt"
        sim_summary_path = output_dir / "sim.summary.txt"
        manifest = read_json(manifest_path)
        asv_mean, asv_std = parse_sim_summary(sim_summary_path)
        metrics = BenchmarkMetrics(
            checkpoint_dir=str(checkpoint_dir.resolve()),
            eval_jsonl=str(eval_jsonl.resolve()),
            output_dir=str(output_dir.resolve()),
            label=str(label),
            wer=float(parse_wer_summary(wer_summary_path)),
            asv_mean=float(asv_mean),
            asv_std=float(asv_std),
            num_samples=int(manifest.get("num_samples", 0)),
            wer_summary_path=str(wer_summary_path.resolve()),
            sim_summary_path=str(sim_summary_path.resolve()),
            manifest_path=str(manifest_path.resolve()),
        )
        write_json(metrics_path, asdict(metrics))
        return metrics

    def _run_generation(self, *, checkpoint_dir: Path, eval_jsonl: Path, output_dir: Path) -> None:
        cmd = [
            self.seed_tts_eval_python,
            str(REPO_ROOT / "finetuning" / "prepare_seed_tts_eval.py"),
            "--checkpoint_dir",
            str(checkpoint_dir),
            "--eval_jsonl",
            str(eval_jsonl),
            "--output_dir",
            str(output_dir),
            "--device",
            str(self.eval_config.benchmark_eval_device),
        ]
        if self.train_config.speaker_name:
            cmd.extend(["--speaker_name", str(self.train_config.speaker_name)])
        self._run(cmd, cwd=REPO_ROOT)

    def _run_scoring(self, *, output_dir: Path) -> None:
        wav_res_ref_text = output_dir / "wav_res_ref_text"
        self._run(
            [
                self.seed_tts_eval_python,
                str(self.seed_tts_eval_root / "get_wav_res_ref_text.py"),
                str(output_dir / "meta.lst"),
                str(output_dir / "generated"),
                str(wav_res_ref_text),
            ],
            cwd=REPO_ROOT,
        )
        self._run(
            [
                self.seed_tts_eval_python,
                str(self.seed_tts_eval_root / "run_wer.py"),
                str(wav_res_ref_text),
                str(output_dir / "wer.raw.txt"),
                "zh",
            ],
            cwd=REPO_ROOT,
        )
        self._run(
            [
                self.seed_tts_eval_python,
                str(self.seed_tts_eval_root / "average_wer.py"),
                str(output_dir / "wer.raw.txt"),
                str(output_dir / "wer.summary.txt"),
            ],
            cwd=REPO_ROOT,
        )
        self._run(
            [
                self.seed_tts_eval_python,
                str(
                    self.seed_tts_eval_root
                    / "thirdparty"
                    / "UniSpeech"
                    / "downstreams"
                    / "speaker_verification"
                    / "verification_pair_list_v2.py"
                ),
                str(wav_res_ref_text),
                "--model_name",
                "wavlm_large",
                "--checkpoint",
                str(self.sim_finetune_checkpoint),
                "--scores",
                str(output_dir / "sim.raw.txt"),
                "--wav1_start_sr",
                "0",
                "--wav2_start_sr",
                "0",
                "--wav1_end_sr",
                "-1",
                "--wav2_end_sr",
                "-1",
                "--device",
                str(self.eval_config.benchmark_eval_device),
            ],
            cwd=REPO_ROOT,
        )
        self._run(
            [
                self.seed_tts_eval_python,
                str(
                    self.seed_tts_eval_root
                    / "thirdparty"
                    / "UniSpeech"
                    / "downstreams"
                    / "speaker_verification"
                    / "average.py"
                ),
                str(output_dir / "sim.raw.txt"),
                str(output_dir / "sim.summary.txt"),
            ],
            cwd=REPO_ROOT,
        )

    def evaluate_checkpoint(self, *, checkpoint_dir: str | Path, eval_jsonl: str | Path, output_dir: str | Path, label: str) -> BenchmarkMetrics:
        checkpoint_dir = _resolve_path(checkpoint_dir, base_dir=REPO_ROOT)
        eval_jsonl = _resolve_path(eval_jsonl, base_dir=REPO_ROOT)
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "benchmark_metrics.json"
        if metrics_path.exists():
            return BenchmarkMetrics(**read_json(metrics_path))

        self.ensure_assets_prepared()
        self._run_generation(checkpoint_dir=checkpoint_dir, eval_jsonl=eval_jsonl, output_dir=output_dir)
        self._run_scoring(output_dir=output_dir)
        return self._build_metrics_from_dir(
            checkpoint_dir=checkpoint_dir,
            eval_jsonl=eval_jsonl,
            output_dir=output_dir,
            label=label,
        )

    def ensure_base_baseline(self, eval_jsonl: str | Path) -> BenchmarkMetrics:
        if self._base_metrics is not None:
            return self._base_metrics
        output_dir = self.paths.benchmark_eval_dir / "base_full"
        self._base_metrics = self.evaluate_checkpoint(
            checkpoint_dir=self.train_config.init_model_path,
            eval_jsonl=eval_jsonl,
            output_dir=output_dir,
            label="base_full",
        )
        return self._base_metrics

    def ensure_benchmark_subset(self, base_metrics: BenchmarkMetrics, eval_jsonl: str | Path):
        if self._subset_path is not None and self._subset_manifest is not None and self._subset_rows is not None:
            return self._subset_rows, self._subset_path, self._subset_manifest

        subset_path = self.paths.benchmark_eval_dir / "benchmark_subset.jsonl"
        subset_manifest_path = self.paths.benchmark_eval_dir / "benchmark_subset_manifest.json"
        if subset_path.exists() and subset_manifest_path.exists():
            self._subset_rows = read_jsonl(subset_path)
            self._subset_path = subset_path
            self._subset_manifest = read_json(subset_manifest_path)
            return self._subset_rows, self._subset_path, self._subset_manifest

        eval_jsonl = _resolve_path(eval_jsonl, base_dir=REPO_ROOT)
        eval_rows = read_jsonl(eval_jsonl)
        base_manifest = read_json(base_metrics.manifest_path)
        base_wer_by_utt = parse_wer_raw(base_metrics.wer_summary_path)
        subset_rows, manifest_rows = derive_benchmark_subset(
            eval_rows=eval_rows,
            base_manifest=base_manifest,
            base_wer_by_utt=base_wer_by_utt,
            sample_count=int(self.eval_config.benchmark_eval_num_samples),
        )
        write_jsonl(subset_path, subset_rows)
        subset_manifest = {
            "source_eval_jsonl": str(eval_jsonl),
            "sample_count": len(subset_rows),
            "base_metrics": asdict(base_metrics),
            "samples": manifest_rows,
        }
        write_json(subset_manifest_path, subset_manifest)
        self._subset_rows = subset_rows
        self._subset_path = subset_path
        self._subset_manifest = subset_manifest
        return self._subset_rows, self._subset_path, self._subset_manifest
