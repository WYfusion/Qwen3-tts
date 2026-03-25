# coding=utf-8

from __future__ import annotations

import time
import uuid


def parse_csv_arg(value: str):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_log_with(value: str):
    names = []
    seen = set()
    for item in parse_csv_arg(value):
        lowered = item.lower()
        if lowered not in seen:
            names.append(lowered)
            seen.add(lowered)
    return names


def default_wandb_group(train_config, dataset_name: str) -> str:
    init_name = train_config.init_model_path.split("/")[-1].split("\\")[-1]
    return f"{train_config.speaker_name}__{dataset_name}__{init_name}"


def default_wandb_run_name(train_config, effective_epochs: int) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return (
        f"{stamp}__recipe-{train_config.training_recipe}"
        f"__bs{train_config.batch_size}x{train_config.gradient_accumulation_steps}"
        f"__ep{effective_epochs}__seed{train_config.seed}"
    )


def wandb_enabled(log_with: list[str]) -> bool:
    return "wandb" in log_with


def get_wandb_run(accelerator):
    try:
        return accelerator.get_tracker("wandb", unwrap=True)
    except Exception:
        return None


def init_trackers(accelerator, *, logging_config, tracker_config: dict, wandb_run_id: str, wandb_run_name: str, wandb_group: str):
    init_kwargs = {}
    if wandb_enabled(parse_log_with(logging_config.log_with_raw)):
        wandb_kwargs = {
            "name": wandb_run_name,
            "group": wandb_group,
            "job_type": logging_config.wandb_job_type,
            "resume": logging_config.wandb_resume,
            "id": wandb_run_id or uuid.uuid4().hex,
            "tags": parse_csv_arg(logging_config.wandb_tags),
        }
        if logging_config.wandb_entity:
            wandb_kwargs["entity"] = logging_config.wandb_entity
        init_kwargs["wandb"] = wandb_kwargs
        wandb_run_id = wandb_kwargs["id"]
    accelerator.init_trackers(
        project_name=logging_config.wandb_project,
        config=tracker_config,
        init_kwargs=init_kwargs,
    )
    run = get_wandb_run(accelerator)
    if run is not None:
        wandb_run_id = str(getattr(run, "id", wandb_run_id))
        wandb_run_name = str(getattr(run, "name", wandb_run_name))
    return wandb_run_id, wandb_run_name

