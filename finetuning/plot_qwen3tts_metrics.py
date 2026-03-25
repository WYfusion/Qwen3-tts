#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def load_csv(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if v is None or v == "":
                    parsed[k] = v
                    continue
                try:
                    parsed[k] = float(v)
                except Exception:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def save_plot(xs, ys, xlabel, ylabel, title, out_path: Path):
    if plt is None or not xs or not ys:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", type=str, required=True)
    parser.add_argument("--plots_dir", type=str, required=True)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    step_csv = metrics_dir / "train_step_metrics.csv"
    epoch_csv = metrics_dir / "train_epoch_metrics.csv"

    if step_csv.exists():
        rows = load_csv(step_csv)
        steps = [int(r["global_step"]) for r in rows]
        for key, name in [
            ("loss", "train_loss_vs_step.png"),
            ("main_loss", "train_main_loss_vs_step.png"),
            ("sub_loss", "train_sub_loss_vs_step.png"),
            ("lr", "learning_rate_vs_step.png"),
            ("grad_norm", "grad_norm_vs_step.png"),
            ("tokens_per_sec", "tokens_per_sec_vs_step.png"),
            ("codec_tokens", "codec_tokens_vs_step.png"),
        ]:
            ys = [float(r[key]) for r in rows if key in r and r[key] != ""]
            if len(ys) == len(steps):
                save_plot(steps, ys, "Global step", key, key, plots_dir / name)

    if epoch_csv.exists():
        rows = load_csv(epoch_csv)
        epochs = [int(r["epoch"]) for r in rows]
        for key, name in [
            ("epoch_loss", "epoch_loss.png"),
            ("epoch_main_loss", "epoch_main_loss.png"),
            ("epoch_sub_loss", "epoch_sub_loss.png"),
            ("epoch_tokens_per_sec", "epoch_tokens_per_sec.png"),
            ("fixed_eval_mean_duration_ratio", "fixed_eval_mean_duration_ratio.png"),
            ("fixed_eval_max_duration_ratio", "fixed_eval_max_duration_ratio.png"),
            ("fixed_eval_mean_abs_duration_error", "fixed_eval_mean_abs_duration_error.png"),
            ("fixed_eval_cap_hit_rate", "fixed_eval_cap_hit_rate.png"),
            ("fixed_eval_qc_score", "fixed_eval_qc_score.png"),
        ]:
            ys = [float(r[key]) for r in rows if key in r and r[key] != ""]
            if len(ys) == len(epochs):
                save_plot(epochs, ys, "Epoch", key, key, plots_dir / name)


if __name__ == "__main__":
    main()
