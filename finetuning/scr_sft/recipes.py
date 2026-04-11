# coding=utf-8

from __future__ import annotations

from torch.optim import AdamW

STAGED_RECIPE_SPECS = {
    "staged_stable_sft": {
        "epochs": ("stage1_epochs", "stage2_epochs"),
        "kd_multipliers": {"stage1": 1.0, "stage2": 1.0},
        "groups": {
            "speaker_row": {
                "prefixes": ["talker.model.codec_embedding.weight"],
                "stage1": {"lr": 1e-6, "trainable": True},
                "stage2": {"lr": 1e-6, "trainable": True},
            },
            "head": {
                "prefixes": ["talker.text_projection.", "talker.codec_head."],
                "stage1": {"lr": 1e-6, "trainable": True},
                "stage2": {"lr": 1e-6, "trainable": True},
            },
            "upper_layers": {
                "prefixes": [
                    "talker.model.layers.24.",
                    "talker.model.layers.25.",
                    "talker.model.layers.26.",
                    "talker.model.layers.27.",
                    "talker.model.norm.",
                ],
                "stage1": {"lr": 8e-7, "trainable": True},
                "stage2": {"lr": 8e-7, "trainable": True},
            },
            "mid_layers": {
                "prefixes": [
                    "talker.model.layers.20.",
                    "talker.model.layers.21.",
                    "talker.model.layers.22.",
                    "talker.model.layers.23.",
                ],
                "stage1": {"lr": 0.0, "trainable": False},
                "stage2": {"lr": 5e-7, "trainable": True},
            },
            "code_predictor": {
                "prefixes": ["talker.code_predictor."],
                "stage1": {"lr": 0.0, "trainable": False},
                "stage2": {"lr": 5e-7, "trainable": True},
            },
        },
        "frozen_summary": [
            "speaker_encoder.*",
            "talker.model.text_embedding.*",
            "talker.model.layers.0-19.*",
        ],
    },
    "staged_benchmark_aligned_sft": {
        "epochs": ("stage1_epochs", "stage2_epochs", "stage3_epochs"),
        "kd_multipliers": {"stage1": 2.0, "stage2": 1.0, "stage3": 0.6},
        "groups": {
            "speaker_row": {
                "prefixes": ["talker.model.codec_embedding.weight"],
                "stage1": {"lr": 5e-6, "trainable": True},
                "stage2": {"lr": 3e-6, "trainable": True},
                "stage3": {"lr": 2e-6, "trainable": True},
            },
            "head": {
                "prefixes": ["talker.text_projection.", "talker.codec_head."],
                "stage1": {"lr": 3e-6, "trainable": True},
                "stage2": {"lr": 2e-6, "trainable": True},
                "stage3": {"lr": 1.5e-6, "trainable": True},
            },
            "upper_layers": {
                "prefixes": [
                    "talker.model.layers.24.",
                    "talker.model.layers.25.",
                    "talker.model.layers.26.",
                    "talker.model.layers.27.",
                    "talker.model.norm.",
                ],
                "stage1": {"lr": 1.5e-6, "trainable": True},
                "stage2": {"lr": 1.0e-6, "trainable": True},
                "stage3": {"lr": 8e-7, "trainable": True},
            },
            "mid_layers": {
                "prefixes": [
                    "talker.model.layers.20.",
                    "talker.model.layers.21.",
                    "talker.model.layers.22.",
                    "talker.model.layers.23.",
                ],
                "stage1": {"lr": 0.0, "trainable": False},
                "stage2": {"lr": 7e-7, "trainable": True},
                "stage3": {"lr": 6e-7, "trainable": True},
            },
            "lower_mid_layers": {
                "prefixes": [
                    "talker.model.layers.16.",
                    "talker.model.layers.17.",
                    "talker.model.layers.18.",
                    "talker.model.layers.19.",
                ],
                "stage1": {"lr": 0.0, "trainable": False},
                "stage2": {"lr": 0.0, "trainable": False},
                "stage3": {"lr": 4e-7, "trainable": True},
            },
            "code_predictor": {
                "prefixes": ["talker.code_predictor."],
                "stage1": {"lr": 0.0, "trainable": False},
                "stage2": {"lr": 7e-7, "trainable": True},
                "stage3": {"lr": 5e-7, "trainable": True},
            },
        },
        "frozen_summary": [
            "speaker_encoder.*",
            "talker.model.text_embedding.*",
            "talker.model.layers.0-15.*",
        ],
    },
}


def is_staged_recipe(recipe_name: str) -> bool:
    return str(recipe_name) in STAGED_RECIPE_SPECS


def recipe_stage_names(train_config) -> list[str]:
    recipe_name = str(train_config.training_recipe)
    if not is_staged_recipe(recipe_name):
        return ["legacy"]
    spec = STAGED_RECIPE_SPECS[recipe_name]
    return [stage_name for stage_name in spec["kd_multipliers"].keys() if int(getattr(train_config, f"{stage_name}_epochs", 0)) > 0]


def recipe_stage_group_specs(recipe_name: str) -> dict:
    if not is_staged_recipe(recipe_name):
        raise KeyError(f"Recipe `{recipe_name}` is not a staged recipe.")
    return STAGED_RECIPE_SPECS[str(recipe_name)]["groups"]


def effective_num_epochs(train_config) -> int:
    if is_staged_recipe(train_config.training_recipe):
        spec = STAGED_RECIPE_SPECS[str(train_config.training_recipe)]
        return sum(int(getattr(train_config, attr_name)) for attr_name in spec["epochs"])
    return int(train_config.num_epochs)


def stage_name_for_epoch(epoch: int, train_config) -> str:
    recipe_name = str(train_config.training_recipe)
    if not is_staged_recipe(recipe_name):
        return "legacy"

    remaining = int(epoch)
    spec = STAGED_RECIPE_SPECS[recipe_name]
    for attr_name in spec["epochs"]:
        stage_name = attr_name.replace("_epochs", "")
        stage_epochs = int(getattr(train_config, attr_name))
        if remaining < stage_epochs:
            return stage_name
        remaining -= stage_epochs
    return spec["epochs"][-1].replace("_epochs", "")


def kd_weights_for_epoch(train_config, epoch: int) -> tuple[float, float]:
    recipe_name = str(train_config.training_recipe)
    if not is_staged_recipe(recipe_name):
        return float(train_config.main_kd_weight), float(train_config.sub_kd_weight)
    stage_name = stage_name_for_epoch(epoch, train_config)
    multiplier = float(STAGED_RECIPE_SPECS[recipe_name]["kd_multipliers"][stage_name])
    return float(train_config.main_kd_weight) * multiplier, float(train_config.sub_kd_weight) * multiplier


def matches_any_prefix(name: str, prefixes: list[str]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def build_optimizer(model, train_config):
    recipe_name = str(train_config.training_recipe)
    if not is_staged_recipe(recipe_name):
        optimizer = AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
        param_names = {"legacy_full_sft": [name for name, _ in model.named_parameters()]}
        return optimizer, {}, param_names

    specs = recipe_stage_group_specs(recipe_name)
    first_stage = stage_name_for_epoch(0, train_config)
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
        param.requires_grad = bool(specs[assigned][first_stage]["trainable"])
        grouped_params[assigned].append(param)
        grouped_param_names[assigned].append(name)

    for group_name, params in grouped_params.items():
        if not params:
            raise RuntimeError(f"Expected non-empty parameter group for staged recipe `{recipe_name}`: {group_name}")

    optimizer_groups = []
    for group_name, params in grouped_params.items():
        optimizer_groups.append(
            {
                "params": params,
                "lr": float(specs[group_name][first_stage]["lr"]),
                "weight_decay": float(train_config.weight_decay),
                "group_name": group_name,
            }
        )
    optimizer = AdamW(optimizer_groups, weight_decay=float(train_config.weight_decay))
    return optimizer, specs, grouped_param_names


def apply_training_stage(model, optimizer, train_config, epoch: int, accelerator, grouped_param_names):
    stage_name = stage_name_for_epoch(epoch, train_config)
    recipe_name = str(train_config.training_recipe)
    if not is_staged_recipe(recipe_name):
        accelerator.print(f"[stage] epoch={epoch} | recipe=legacy_full_sft | all parameters trainable")
        return stage_name

    specs = recipe_stage_group_specs(recipe_name)
    for name, param in model.named_parameters():
        assigned = None
        for group_name, spec in specs.items():
            if matches_any_prefix(name, spec["prefixes"]):
                assigned = group_name
                break
        if assigned is None:
            param.requires_grad = False
            continue
        param.requires_grad = bool(specs[assigned][stage_name]["trainable"])

    for param_group in optimizer.param_groups:
        group_name = param_group.get("group_name")
        if group_name not in specs:
            continue
        param_group["lr"] = float(specs[group_name][stage_name]["lr"])

    accelerator.print(f"[stage] epoch={epoch} | recipe={recipe_name} | switched to {stage_name}")
    for group_name, names in grouped_param_names.items():
        group_spec = specs[group_name][stage_name]
        accelerator.print(
            f"  - group={group_name} | trainable={group_spec['trainable']} | "
            f"lr={group_spec['lr']:.2e} | params={len(names)}"
        )
    accelerator.print("  - frozen: " + ", ".join(STAGED_RECIPE_SPECS[recipe_name]["frozen_summary"]))
    return stage_name


def register_speaker_row_gradient_mask(model, custom_speaker_id: int):
    embedding_weight = model.talker.model.codec_embedding.weight

    def _mask_non_speaker_rows(grad):
        masked = grad.new_zeros(grad.shape)
        masked[custom_speaker_id] = grad[custom_speaker_id]
        return masked

    return embedding_weight.register_hook(_mask_non_speaker_rows)
