# coding=utf-8

from __future__ import annotations

from torch.optim import AdamW


def effective_num_epochs(train_config) -> int:
    if train_config.training_recipe == "staged_stable_sft":
        return int(train_config.stage1_epochs) + int(train_config.stage2_epochs)
    return int(train_config.num_epochs)


def stage_name_for_epoch(epoch: int, train_config) -> str:
    if train_config.training_recipe != "staged_stable_sft":
        return "legacy"
    if epoch < int(train_config.stage1_epochs):
        return "stage1"
    return "stage2"


def matches_any_prefix(name: str, prefixes: list[str]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def build_stage_group_specs():
    return {
        "speaker_row": {
            "prefixes": ["talker.model.codec_embedding.weight"],
            "stage1_lr": 1e-6,
            "stage2_lr": 1e-6,
            "stage1_trainable": True,
            "stage2_trainable": True,
        },
        "head": {
            "prefixes": ["talker.text_projection.", "talker.codec_head."],
            "stage1_lr": 1e-6,
            "stage2_lr": 1e-6,
            "stage1_trainable": True,
            "stage2_trainable": True,
        },
        "upper_layers": {
            "prefixes": [
                "talker.model.layers.24.",
                "talker.model.layers.25.",
                "talker.model.layers.26.",
                "talker.model.layers.27.",
                "talker.model.norm.",
            ],
            "stage1_lr": 8e-7,
            "stage2_lr": 8e-7,
            "stage1_trainable": True,
            "stage2_trainable": True,
        },
        "mid_layers": {
            "prefixes": [
                "talker.model.layers.20.",
                "talker.model.layers.21.",
                "talker.model.layers.22.",
                "talker.model.layers.23.",
            ],
            "stage1_lr": 0.0,
            "stage2_lr": 5e-7,
            "stage1_trainable": False,
            "stage2_trainable": True,
        },
        "code_predictor": {
            "prefixes": ["talker.code_predictor."],
            "stage1_lr": 0.0,
            "stage2_lr": 5e-7,
            "stage1_trainable": False,
            "stage2_trainable": True,
        },
    }


def build_optimizer(model, train_config):
    if train_config.training_recipe != "staged_stable_sft":
        optimizer = AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
        param_names = {"legacy_full_sft": [name for name, _ in model.named_parameters()]}
        return optimizer, {}, param_names

    specs = build_stage_group_specs()
    grouped_params = {group_name: [] for group_name in specs}
    grouped_param_names = {group_name: [] for group_name in specs}
    for name, param in model.named_parameters():
        assigned = None
        for group_name, spec in specs.items():
            if matches_any_prefix(name, spec["prefixes"]):
                assigned = group_name
                break
        if assigned is None:
            param.requires_grad = False
            continue
        param.requires_grad = bool(specs[assigned]["stage1_trainable"])
        grouped_params[assigned].append(param)
        grouped_param_names[assigned].append(name)

    for group_name, params in grouped_params.items():
        if not params:
            raise RuntimeError(f"Expected non-empty parameter group for staged recipe: {group_name}")

    optimizer_groups = []
    for group_name, params in grouped_params.items():
        optimizer_groups.append(
            {
                "params": params,
                "lr": float(specs[group_name]["stage1_lr"]),
                "weight_decay": float(train_config.weight_decay),
                "group_name": group_name,
            }
        )
    optimizer = AdamW(optimizer_groups, weight_decay=float(train_config.weight_decay))
    return optimizer, specs, grouped_param_names


def apply_training_stage(model, optimizer, train_config, epoch: int, accelerator, grouped_param_names):
    stage_name = stage_name_for_epoch(epoch, train_config)
    if train_config.training_recipe != "staged_stable_sft":
        accelerator.print(f"[stage] epoch={epoch} | recipe=legacy_full_sft | all parameters trainable")
        return stage_name

    specs = build_stage_group_specs()
    for name, param in model.named_parameters():
        assigned = None
        for group_name, spec in specs.items():
            if matches_any_prefix(name, spec["prefixes"]):
                assigned = group_name
                break
        if assigned is None:
            param.requires_grad = False
            continue
        trainable_key = "stage1_trainable" if stage_name == "stage1" else "stage2_trainable"
        param.requires_grad = bool(specs[assigned][trainable_key])

    for param_group in optimizer.param_groups:
        group_name = param_group.get("group_name")
        if group_name not in specs:
            continue
        lr_key = "stage1_lr" if stage_name == "stage1" else "stage2_lr"
        param_group["lr"] = float(specs[group_name][lr_key])

    accelerator.print(f"[stage] epoch={epoch} | switched to {stage_name}")
    for group_name, names in grouped_param_names.items():
        lr_key = "stage1_lr" if stage_name == "stage1" else "stage2_lr"
        trainable_key = "stage1_trainable" if stage_name == "stage1" else "stage2_trainable"
        accelerator.print(
            f"  - group={group_name} | trainable={specs[group_name][trainable_key]} | "
            f"lr={specs[group_name][lr_key]:.2e} | params={len(names)}"
        )
    accelerator.print("  - frozen: speaker_encoder.*, talker.model.layers.0-19.*, talker.model.text_embedding.*")
    if stage_name == "stage1":
        accelerator.print("  - additionally frozen: talker.model.layers.20-23.*, talker.code_predictor.*")
    return stage_name


def register_speaker_row_gradient_mask(model, custom_speaker_id: int):
    embedding_weight = model.talker.model.codec_embedding.weight

    def _mask_non_speaker_rows(grad):
        masked = grad.new_zeros(grad.shape)
        masked[custom_speaker_id] = grad[custom_speaker_id]
        return masked

    return embedding_weight.register_hook(_mask_non_speaker_rows)

