# coding=utf-8

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .io_utils import ensure_dir


@dataclass
class TrainConfig:
    init_model_path: str
    output_model_path: str
    train_jsonl: str
    batch_size: int
    lr: float
    num_epochs: int
    speaker_name: str
    gradient_accumulation_steps: int
    weight_decay: float
    max_grad_norm: float
    warmup_ratio: float
    log_steps: int
    save_plots_every_epoch: bool
    disable_flash_attn: bool
    seed: int
    training_recipe: str
    stage1_epochs: int
    stage2_epochs: int
    main_kd_weight: float
    sub_kd_weight: float
    kd_temperature: float
    audio_qc_report_only: bool
    dry_run: bool


@dataclass
class EvalConfig:
    fixed_eval_jsonl: str | None
    fixed_eval_source_jsonl: str | None
    fixed_eval_num_samples: int
    fixed_eval_length_mode: str
    fixed_eval_length_multiplier: float
    fixed_eval_max_new_tokens: int
    fixed_eval_language: str
    fixed_eval_do_sample: bool
    fixed_eval_duration_ratio_warn: float
    enable_free_run_eval: bool
    free_run_eval_max_new_tokens: int
    peak_warn_threshold: float
    clipped_frac_warn_threshold: float
    hf_noise_warn_threshold: float
    voiced_f0_delta_warn_threshold: float
    skip_fixed_eval_generation: bool


@dataclass
class LoggingConfig:
    log_with_raw: str
    wandb_project: str
    wandb_entity: str | None
    wandb_group: str | None
    wandb_run_name: str | None
    wandb_tags: str
    wandb_job_type: str
    wandb_resume: str


@dataclass
class CheckpointConfig:
    resume_from_training_state: str | None
    save_training_state_steps: int
    keep_last_training_states: int
    early_stop_patience: int


@dataclass
class RunPaths:
    output_dir: Path
    logs_dir: Path
    tb_dir: Path
    metrics_dir: Path
    plots_dir: Path
    anomaly_dir: Path
    fixed_eval_audio_dir: Path
    free_run_eval_audio_dir: Path
    qc_report_dir: Path
    training_state_dir: Path
    step_csv: Path
    epoch_csv: Path

    @classmethod
    def from_output_root(cls, output_root: str | Path) -> "RunPaths":
        output_dir = ensure_dir(output_root)
        logs_dir = ensure_dir(output_dir / "logs")
        metrics_dir = ensure_dir(logs_dir / "metrics")
        return cls(
            output_dir=output_dir,
            logs_dir=logs_dir,
            tb_dir=ensure_dir(logs_dir / "tensorboard"),
            metrics_dir=metrics_dir,
            plots_dir=ensure_dir(logs_dir / "plots"),
            anomaly_dir=ensure_dir(logs_dir / "anomaly_batches"),
            fixed_eval_audio_dir=ensure_dir(logs_dir / "fixed_eval_audio"),
            free_run_eval_audio_dir=ensure_dir(logs_dir / "free_run_eval_audio"),
            qc_report_dir=ensure_dir(logs_dir / "audio_qc"),
            training_state_dir=ensure_dir(output_dir / "training_state"),
            step_csv=metrics_dir / "train_step_metrics.csv",
            epoch_csv=metrics_dir / "train_epoch_metrics.csv",
        )


@dataclass
class SFTConfig:
    train: TrainConfig
    eval: EvalConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig
    paths: RunPaths

