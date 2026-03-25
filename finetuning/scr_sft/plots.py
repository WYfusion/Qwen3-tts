# coding=utf-8

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .constants import EPOCH_PLOT_KEYS, STEP_PLOT_KEYS
from .io_utils import ensure_dir

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


class MetricsPlotter:
    def __init__(self, save_dir: Path):
        self.save_dir = ensure_dir(save_dir)

    def _load_csv(self, csv_path: Path):
        rows = []
        if not csv_path.exists():
            return rows
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {}
                for key, value in row.items():
                    if value in (None, ""):
                        parsed[key] = value
                        continue
                    try:
                        parsed[key] = float(value)
                    except Exception:
                        parsed[key] = value
                rows.append(parsed)
        return rows

    def _save_line_plot(self, xs, ys, xlabel: str, ylabel: str, title: str, save_path: Path):
        if plt is None or not xs or not ys:
            return
        plt.figure(figsize=(10, 6))
        plt.plot(xs, ys)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        plt.close()

    def plot_step_metrics(self, step_csv: Path):
        rows = self._load_csv(step_csv)
        if not rows:
            return
        steps = [int(row["global_step"]) for row in rows]
        for key, filename in STEP_PLOT_KEYS:
            ys = [float(row[key]) for row in rows if key in row and row[key] != ""]
            if len(ys) == len(steps):
                self._save_line_plot(steps, ys, "Global step", key, key, self.save_dir / filename)

    def plot_epoch_metrics(self, epoch_csv: Path):
        rows = self._load_csv(epoch_csv)
        if not rows:
            return
        epochs = [int(row["epoch"]) for row in rows]
        for key, filename in EPOCH_PLOT_KEYS:
            ys = [float(row[key]) for row in rows if key in row and row[key] != ""]
            if len(ys) == len(epochs):
                self._save_line_plot(epochs, ys, "Epoch", key, key, self.save_dir / filename)

    def plot_all(self, *, metrics_dir: Path):
        self.plot_step_metrics(metrics_dir / "train_step_metrics.csv")
        self.plot_epoch_metrics(metrics_dir / "train_epoch_metrics.csv")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", type=str, required=True)
    parser.add_argument("--plots_dir", type=str, required=True)
    args = parser.parse_args(argv)

    plotter = MetricsPlotter(Path(args.plots_dir))
    plotter.plot_all(metrics_dir=Path(args.metrics_dir))


if __name__ == "__main__":
    main()
