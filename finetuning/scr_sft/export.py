# coding=utf-8

from __future__ import annotations

import json
import shutil
from pathlib import Path

from safetensors.torch import save_file

from .constants import CUSTOM_SPEAKER_ID
from .io_utils import write_json


def export_inference_checkpoint(
    *,
    accelerator,
    model,
    model_path: str,
    output_model_path: Path,
    epoch: int,
    speaker_name: str,
    summary: dict,
):
    ckpt_dir = output_model_path / f"checkpoint-epoch-{epoch}"
    shutil.copytree(model_path, ckpt_dir, dirs_exist_ok=True)

    input_config_file = Path(model_path) / "config.json"
    output_config_file = ckpt_dir / "config.json"
    with input_config_file.open("r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict["tts_model_type"] = "custom_voice"
    talker_config = config_dict.get("talker_config", {})
    talker_config["spk_id"] = {speaker_name: CUSTOM_SPEAKER_ID}
    talker_config["spk_is_dialect"] = {speaker_name: False}
    config_dict["talker_config"] = talker_config
    with output_config_file.open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}
    keys_to_drop = [k for k in state_dict.keys() if k.startswith("speaker_encoder")]
    for key in keys_to_drop:
        del state_dict[key]
    save_file(state_dict, ckpt_dir / "model.safetensors")
    write_json(ckpt_dir / "train_summary.json", summary)
    return ckpt_dir


def write_best_checkpoint_record(
    output_dir: Path,
    *,
    best_epoch: int,
    best_checkpoint_path: str,
    best_qc_score: float,
    best_eval_name: str,
):
    payload = {
        "best_epoch": int(best_epoch),
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_qc_score": float(best_qc_score),
        "best_eval_name": str(best_eval_name),
    }
    write_json(output_dir / "best_checkpoint.json", payload)
    return payload
