# coding=utf-8

from __future__ import annotations

import shutil
from pathlib import Path

import torch

from .io_utils import detach_to_cpu, read_json, write_json


class TrainingStateTracker:
    def __init__(self):
        self.next_epoch = 0
        self.next_batch_in_epoch = 0
        self.global_step = 0
        self.last_step_checkpoint = ""
        self.last_reason = ""
        self.wandb_run_id = ""
        self.wandb_run_name = ""
        self.best_qc_score = None
        self.best_checkpoint_path = ""
        self.best_epoch = -1
        self.best_eval_name = ""
        self.epochs_since_improvement = 0

    def mark_progress(self, *, next_epoch: int, next_batch_in_epoch: int, global_step: int, last_step_checkpoint: str = "", last_reason: str = "", wandb_run_id: str | None = None, wandb_run_name: str | None = None, best_qc_score=None, best_checkpoint_path: str | None = None, best_epoch: int | None = None, best_eval_name: str | None = None, epochs_since_improvement: int | None = None):
        self.next_epoch = int(next_epoch)
        self.next_batch_in_epoch = int(next_batch_in_epoch)
        self.global_step = int(global_step)
        self.last_step_checkpoint = str(last_step_checkpoint)
        self.last_reason = str(last_reason)
        if wandb_run_id is not None:
            self.wandb_run_id = str(wandb_run_id)
        if wandb_run_name is not None:
            self.wandb_run_name = str(wandb_run_name)
        if best_qc_score is not None:
            self.best_qc_score = float(best_qc_score)
        if best_checkpoint_path is not None:
            self.best_checkpoint_path = str(best_checkpoint_path)
        if best_epoch is not None:
            self.best_epoch = int(best_epoch)
        if best_eval_name is not None:
            self.best_eval_name = str(best_eval_name)
        if epochs_since_improvement is not None:
            self.epochs_since_improvement = int(epochs_since_improvement)

    def state_dict(self):
        return {
            "next_epoch": self.next_epoch,
            "next_batch_in_epoch": self.next_batch_in_epoch,
            "global_step": self.global_step,
            "last_step_checkpoint": self.last_step_checkpoint,
            "last_reason": self.last_reason,
            "wandb_run_id": self.wandb_run_id,
            "wandb_run_name": self.wandb_run_name,
            "best_qc_score": self.best_qc_score,
            "best_checkpoint_path": self.best_checkpoint_path,
            "best_epoch": self.best_epoch,
            "best_eval_name": self.best_eval_name,
            "epochs_since_improvement": self.epochs_since_improvement,
        }

    def load_state_dict(self, state_dict):
        self.next_epoch = int(state_dict.get("next_epoch", 0))
        self.next_batch_in_epoch = int(state_dict.get("next_batch_in_epoch", 0))
        self.global_step = int(state_dict.get("global_step", 0))
        self.last_step_checkpoint = str(state_dict.get("last_step_checkpoint", ""))
        self.last_reason = str(state_dict.get("last_reason", ""))
        self.wandb_run_id = str(state_dict.get("wandb_run_id", ""))
        self.wandb_run_name = str(state_dict.get("wandb_run_name", ""))
        self.best_qc_score = None if state_dict.get("best_qc_score") in (None, "") else float(state_dict.get("best_qc_score"))
        self.best_checkpoint_path = str(state_dict.get("best_checkpoint_path", ""))
        self.best_epoch = int(state_dict.get("best_epoch", -1))
        self.best_eval_name = str(state_dict.get("best_eval_name", ""))
        self.epochs_since_improvement = int(state_dict.get("epochs_since_improvement", 0))


def load_training_state_metadata(resume_checkpoint: Path | None):
    if resume_checkpoint is None:
        return {}
    metadata_path = resume_checkpoint / "training_state.json"
    if metadata_path.exists():
        return read_json(metadata_path)
    return {}


def latest_step_checkpoint(training_state_dir: Path) -> Path | None:
    if not training_state_dir.exists():
        return None
    candidates = [p for p in training_state_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-step-")]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def resolve_resume_checkpoint(resume_arg: str | None, training_state_dir: Path) -> Path | None:
    if not resume_arg:
        return None
    if resume_arg.lower() == "latest":
        return latest_step_checkpoint(training_state_dir)
    path = Path(resume_arg)
    if path.is_dir():
        return path
    raise ValueError(f"Resume checkpoint not found: {resume_arg}")


def save_training_state_checkpoint(accelerator, tracker: TrainingStateTracker, training_state_dir: Path, keep_last: int, reason: str):
    ckpt_dir = training_state_dir / f"checkpoint-step-{tracker.global_step:08d}"
    tracker.last_step_checkpoint = str(ckpt_dir)
    tracker.last_reason = reason
    accelerator.wait_for_everyone()
    accelerator.save_state(output_dir=str(ckpt_dir))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        metadata = tracker.state_dict()
        metadata["reason"] = reason
        metadata["checkpoint_dir"] = str(ckpt_dir)
        write_json(ckpt_dir / "training_state.json", metadata)
        write_json(training_state_dir / "latest.json", metadata)
        checkpoints = sorted(
            [p for p in training_state_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-step-")],
            key=lambda p: p.name,
        )
        while keep_last > 0 and len(checkpoints) > keep_last:
            stale = checkpoints.pop(0)
            shutil.rmtree(stale, ignore_errors=True)
    accelerator.wait_for_everyone()
    return ckpt_dir


def dump_anomaly_batch(anomaly_dir: Path, *, epoch: int, step: int, global_step: int, reason: str, batch, exception_text: str, latest_step_checkpoint: str):
    dump_dir = anomaly_dir / f"epoch-{epoch:04d}_step-{step:06d}_global-{global_step:08d}"
    dump_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "reason": reason,
            "latest_step_checkpoint": latest_step_checkpoint,
            "batch": detach_to_cpu(batch),
        },
        dump_dir / "batch.pt",
    )
    write_json(
        dump_dir / "metadata.json",
        {
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "reason": reason,
            "latest_step_checkpoint": latest_step_checkpoint,
            "exception": exception_text,
            "sample_meta": batch.get("sample_meta", []),
        },
    )
    return dump_dir
