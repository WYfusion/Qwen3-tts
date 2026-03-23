# coding=utf-8
import argparse
import csv
import json
import math
import os
import shutil
import time
import traceback
from pathlib import Path

import torch
from accelerate import Accelerator, skip_first_batches
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import AutoConfig, get_cosine_schedule_with_warmup

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


target_speaker_embedding = None


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, ensure_ascii=False, indent=2)


def _write_jsonl(path: str | Path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_ready(row), ensure_ascii=False) + "\n")


def _json_ready(obj):
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
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    return str(obj)


def append_metrics_csv(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    fieldnames = list(row.keys())
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _safe_round(value, digits=8):
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return round(value, digits)


def _detach_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _detach_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_detach_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_to_cpu(v) for v in obj)
    return obj


def _sanitize_filename(name: str, max_len: int = 48) -> str:
    cleaned = []
    for ch in name.strip():
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in {"-", "_"}:
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append("_")
    value = "".join(cleaned).strip("_")
    return value[:max_len] if value else "sample"


def _extract_target_embedding(model, ref_mels):
    return model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()


class MetricsPlotter:
    def __init__(self, save_dir: Path):
        self.save_dir = _ensure_dir(save_dir)

    def _load_csv(self, csv_path: Path):
        rows = []
        if not csv_path.exists():
            return rows
        with csv_path.open("r", encoding="utf-8") as f:
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

    def _save_line_plot(self, xs, ys, xlabel, ylabel, title, save_path: Path):
        if plt is None or len(xs) == 0 or len(ys) == 0:
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
                self._save_line_plot(steps, ys, "Global step", key, key, self.save_dir / name)

    def plot_epoch_metrics(self, epoch_csv: Path):
        rows = self._load_csv(epoch_csv)
        if not rows:
            return
        epochs = [int(r["epoch"]) for r in rows]
        for key, name in [
            ("epoch_loss", "epoch_loss.png"),
            ("epoch_main_loss", "epoch_main_loss.png"),
            ("epoch_sub_loss", "epoch_sub_loss.png"),
            ("epoch_tokens_per_sec", "epoch_tokens_per_sec.png"),
        ]:
            ys = [float(r[key]) for r in rows if key in r and r[key] != ""]
            if len(ys) == len(epochs):
                self._save_line_plot(epochs, ys, "Epoch", key, key, self.save_dir / name)


class TrainingStateTracker:
    def __init__(self):
        self.next_epoch = 0
        self.next_batch_in_epoch = 0
        self.global_step = 0
        self.target_speaker_embedding = None
        self.last_step_checkpoint = ""
        self.last_reason = ""

    def mark_progress(
        self,
        *,
        next_epoch: int,
        next_batch_in_epoch: int,
        global_step: int,
        target_speaker_embedding,
        last_step_checkpoint: str = "",
        last_reason: str = "",
    ) -> None:
        self.next_epoch = int(next_epoch)
        self.next_batch_in_epoch = int(next_batch_in_epoch)
        self.global_step = int(global_step)
        self.target_speaker_embedding = (
            None if target_speaker_embedding is None else target_speaker_embedding.detach().cpu()
        )
        self.last_step_checkpoint = str(last_step_checkpoint)
        self.last_reason = str(last_reason)

    def state_dict(self):
        return {
            "next_epoch": self.next_epoch,
            "next_batch_in_epoch": self.next_batch_in_epoch,
            "global_step": self.global_step,
            "target_speaker_embedding": self.target_speaker_embedding,
            "last_step_checkpoint": self.last_step_checkpoint,
            "last_reason": self.last_reason,
        }

    def load_state_dict(self, state_dict):
        self.next_epoch = int(state_dict.get("next_epoch", 0))
        self.next_batch_in_epoch = int(state_dict.get("next_batch_in_epoch", 0))
        self.global_step = int(state_dict.get("global_step", 0))
        self.target_speaker_embedding = state_dict.get("target_speaker_embedding")
        self.last_step_checkpoint = str(state_dict.get("last_step_checkpoint", ""))
        self.last_reason = str(state_dict.get("last_reason", ""))


def _build_epoch_dataloader(dataset, batch_size, collate_fn, epoch: int, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    epoch_subset = Subset(dataset, indices)
    return DataLoader(epoch_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def _latest_step_checkpoint(training_state_dir: Path) -> Path | None:
    if not training_state_dir.exists():
        return None
    candidates = [p for p in training_state_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-step-")]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def _resolve_resume_checkpoint(resume_arg: str | None, training_state_dir: Path) -> Path | None:
    if not resume_arg:
        return None
    if resume_arg.lower() == "latest":
        return _latest_step_checkpoint(training_state_dir)
    path = Path(resume_arg)
    if path.is_dir():
        return path
    raise ValueError(f"Resume checkpoint not found: {resume_arg}")


def _save_training_state_checkpoint(
    accelerator: Accelerator,
    tracker: TrainingStateTracker,
    training_state_dir: Path,
    keep_last: int,
    reason: str,
):
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
        _write_json(ckpt_dir / "training_state.json", metadata)
        _write_json(training_state_dir / "latest.json", metadata)
        checkpoints = sorted(
            [p for p in training_state_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-step-")],
            key=lambda p: p.name,
        )
        while keep_last > 0 and len(checkpoints) > keep_last:
            stale = checkpoints.pop(0)
            shutil.rmtree(stale, ignore_errors=True)
    accelerator.wait_for_everyone()
    return ckpt_dir


def _load_or_create_fixed_eval_set(args, train_data, logs_dir: Path):
    if args.fixed_eval_num_samples <= 0:
        return [], None
    eval_path = Path(args.fixed_eval_jsonl) if args.fixed_eval_jsonl else logs_dir / "fixed_eval_set.jsonl"
    if eval_path.exists():
        return _read_jsonl(eval_path), eval_path
    fixed_eval_rows = []
    for item in train_data[: args.fixed_eval_num_samples]:
        fixed_eval_rows.append(
            {
                "text": item["text"],
                "language": item.get("language", "Auto"),
                "instruct": item.get("instruct", ""),
                "speaker": args.speaker_name,
            }
        )
    _write_jsonl(eval_path, fixed_eval_rows)
    return fixed_eval_rows, eval_path


def _dump_anomaly_batch(
    anomaly_dir: Path,
    *,
    epoch: int,
    step: int,
    global_step: int,
    reason: str,
    batch,
    exception_text: str,
    latest_step_checkpoint: str,
):
    dump_dir = anomaly_dir / f"epoch-{epoch:04d}_step-{step:06d}_global-{global_step:08d}"
    dump_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "reason": reason,
            "latest_step_checkpoint": latest_step_checkpoint,
            "batch": _detach_to_cpu(batch),
        },
        dump_dir / "batch.pt",
    )
    _write_json(
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


def _render_fixed_eval_samples(
    *,
    accelerator: Accelerator,
    model,
    processor,
    speaker_name: str,
    target_embedding,
    fixed_eval_rows,
    output_dir: Path,
    max_new_tokens: int,
):
    if not accelerator.is_main_process or not fixed_eval_rows or target_embedding is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    previous_mode = unwrapped_model.training
    previous_tts_model_type = unwrapped_model.tts_model_type
    previous_config_tts_model_type = unwrapped_model.config.tts_model_type
    previous_spk_id = dict(unwrapped_model.config.talker_config.spk_id)
    previous_spk_is_dialect = dict(unwrapped_model.config.talker_config.spk_is_dialect)
    previous_supported_speakers = unwrapped_model.supported_speakers
    embedding_weight = unwrapped_model.talker.model.codec_embedding.weight
    original_embedding = embedding_weight[3000].detach().clone()

    with torch.no_grad():
        embedding_weight[3000].copy_(target_embedding[0].to(device=embedding_weight.device, dtype=embedding_weight.dtype))
        unwrapped_model.config.tts_model_type = "custom_voice"
        unwrapped_model.tts_model_type = "custom_voice"
        unwrapped_model.config.talker_config.spk_id = {speaker_name: 3000}
        unwrapped_model.config.talker_config.spk_is_dialect = {speaker_name: False}
        unwrapped_model.supported_speakers = unwrapped_model.config.talker_config.spk_id.keys()

    try:
        import soundfile as sf

        unwrapped_model.eval()
        tts = Qwen3TTSModel(unwrapped_model, processor, generate_defaults=unwrapped_model.generate_config)
        manifest = []
        for idx, row in enumerate(fixed_eval_rows):
            if accelerator.is_main_process:
                accelerator.print(
                    f"[fixed-eval] sample {idx + 1}/{len(fixed_eval_rows)} | "
                    f"text_len={len(row['text'])} | max_new_tokens={max_new_tokens}"
                )
            sample_start = time.perf_counter()
            wavs, sample_rate = tts.generate_custom_voice(
                text=row["text"],
                language=row.get("language", "Auto"),
                speaker=row.get("speaker", speaker_name),
                instruct=row.get("instruct", ""),
                max_new_tokens=max_new_tokens,
            )
            stem = _sanitize_filename(row["text"])
            wav_path = output_dir / f"{idx:02d}_{stem}.wav"
            sf.write(wav_path, wavs[0], sample_rate)
            manifest.append(
                {
                    "index": idx,
                    "wav_path": str(wav_path),
                    "text": row["text"],
                    "language": row.get("language", "Auto"),
                    "instruct": row.get("instruct", ""),
                    "speaker": row.get("speaker", speaker_name),
                    "sample_rate": sample_rate,
                    "seconds": round(time.perf_counter() - sample_start, 4),
                }
            )
        _write_json(output_dir / "manifest.json", {"samples": manifest})
    finally:
        with torch.no_grad():
            embedding_weight[3000].copy_(original_embedding)
            unwrapped_model.config.tts_model_type = previous_config_tts_model_type
            unwrapped_model.tts_model_type = previous_tts_model_type
            unwrapped_model.config.talker_config.spk_id = previous_spk_id
            unwrapped_model.config.talker_config.spk_is_dialect = previous_spk_is_dialect
            unwrapped_model.supported_speakers = previous_supported_speakers
        if previous_mode:
            unwrapped_model.train()


def _export_inference_checkpoint(
    *,
    accelerator: Accelerator,
    model,
    model_path: str,
    output_model_path: Path,
    epoch: int,
    speaker_name: str,
    target_embedding,
    summary: dict,
):
    ckpt_dir = output_model_path / f"checkpoint-epoch-{epoch}"
    shutil.copytree(model_path, ckpt_dir, dirs_exist_ok=True)

    input_config_file = os.path.join(model_path, "config.json")
    output_config_file = ckpt_dir / "config.json"
    with open(input_config_file, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict["tts_model_type"] = "custom_voice"
    talker_config = config_dict.get("talker_config", {})
    talker_config["spk_id"] = {speaker_name: 3000}
    talker_config["spk_is_dialect"] = {speaker_name: False}
    config_dict["talker_config"] = talker_config

    with output_config_file.open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}
    keys_to_drop = [k for k in state_dict.keys() if k.startswith("speaker_encoder")]
    for key in keys_to_drop:
        del state_dict[key]

    if target_embedding is None:
        raise RuntimeError("target_speaker_embedding is still None, cannot export inference checkpoint.")

    weight = state_dict["talker.model.codec_embedding.weight"]
    state_dict["talker.model.codec_embedding.weight"][3000] = target_embedding[0].detach().to(weight.device).to(weight.dtype)
    save_file(state_dict, ckpt_dir / "model.safetensors")
    _write_json(ckpt_dir / "train_summary.json", summary)
    return ckpt_dir


def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="./Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_plots_every_epoch", action="store_true")
    parser.add_argument("--disable_flash_attn", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_training_state", type=str, default=None)
    parser.add_argument("--save_training_state_steps", type=int, default=100)
    parser.add_argument("--keep_last_training_states", type=int, default=5)
    parser.add_argument("--fixed_eval_jsonl", type=str, default=None)
    parser.add_argument("--fixed_eval_num_samples", type=int, default=4)
    parser.add_argument("--fixed_eval_max_new_tokens", type=int, default=1536)
    parser.add_argument("--skip_fixed_eval_generation", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = _ensure_dir(args.output_model_path)
    logs_dir = _ensure_dir(output_dir / "logs")
    tb_dir = _ensure_dir(logs_dir / "tensorboard")
    metrics_dir = _ensure_dir(logs_dir / "metrics")
    plots_dir = _ensure_dir(logs_dir / "plots")
    anomaly_dir = _ensure_dir(logs_dir / "anomaly_batches")
    fixed_eval_audio_dir = _ensure_dir(logs_dir / "fixed_eval_audio")
    training_state_dir = _ensure_dir(output_dir / "training_state")
    step_csv = metrics_dir / "train_step_metrics.csv"
    epoch_csv = metrics_dir / "train_epoch_metrics.csv"
    plotter = MetricsPlotter(plots_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=str(tb_dir),
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(project_name="qwen3_tts_sft", config=vars(args))

    model_path = args.init_model_path
    attn_impl = None if args.disable_flash_attn else "flash_attention_2"

    qwen3tts = Qwen3TTSModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    config = AutoConfig.from_pretrained(model_path)
    train_data = _read_jsonl(args.train_jsonl)
    fixed_eval_rows, fixed_eval_path = _load_or_create_fixed_eval_set(args, train_data, logs_dir)
    dataset = TTSDataset(train_data, qwen3tts.processor, config)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(dataset) / args.batch_size / args.gradient_accumulation_steps)
    total_training_steps = max(1, args.num_epochs * steps_per_epoch)
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    model, optimizer, scheduler = accelerator.prepare(qwen3tts.model, optimizer, scheduler)
    tracker = TrainingStateTracker()
    accelerator.register_for_checkpointing(tracker)

    resume_checkpoint = _resolve_resume_checkpoint(args.resume_from_training_state, training_state_dir)
    if resume_checkpoint is not None:
        accelerator.load_state(str(resume_checkpoint))
        target_speaker_embedding = tracker.target_speaker_embedding
        accelerator.print(
            f"Resumed from {resume_checkpoint} | next_epoch={tracker.next_epoch} "
            f"| next_batch_in_epoch={tracker.next_batch_in_epoch} | global_step={tracker.global_step}"
        )

    start_epoch = tracker.next_epoch
    global_step = tracker.global_step
    latest_step_checkpoint = tracker.last_step_checkpoint
    model.train()

    for epoch in range(start_epoch, args.num_epochs):
        raw_epoch_dataloader = _build_epoch_dataloader(dataset, args.batch_size, dataset.collate_fn, epoch, args.seed)
        train_dataloader = accelerator.prepare_data_loader(raw_epoch_dataloader)
        resume_batches = tracker.next_batch_in_epoch if epoch == start_epoch else 0
        if resume_batches > 0:
            train_dataloader = skip_first_batches(train_dataloader, resume_batches)

        epoch_loss_sum = 0.0
        epoch_main_loss_sum = 0.0
        epoch_sub_loss_sum = 0.0
        epoch_logged_steps = 0
        epoch_token_count = 0
        pending_loss_sum = 0.0
        pending_main_loss_sum = 0.0
        pending_sub_loss_sum = 0.0
        pending_micro_steps = 0
        pending_token_count = 0
        sync_window_start = time.perf_counter()
        epoch_wall_start = time.perf_counter()
        effective_step = resume_batches - 1

        for batch in train_dataloader:
            effective_step += 1
            try:
                with accelerator.accumulate(model):
                    input_ids = batch["input_ids"]
                    codec_ids = batch["codec_ids"]
                    ref_mels = batch["ref_mels"]
                    text_embedding_mask = batch["text_embedding_mask"]
                    codec_embedding_mask = batch["codec_embedding_mask"]
                    attention_mask = batch["attention_mask"]
                    codec_0_labels = batch["codec_0_labels"]
                    codec_mask = batch["codec_mask"]

                    speaker_embedding = _extract_target_embedding(model, ref_mels)
                    if target_speaker_embedding is None:
                        target_speaker_embedding = speaker_embedding.detach().cpu()

                    input_text_ids = input_ids[:, :, 0]
                    input_codec_ids = input_ids[:, :, 1]
                    input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                    input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                    input_codec_embedding[:, 6, :] = speaker_embedding

                    input_embeddings = input_text_embedding + input_codec_embedding
                    for i in range(1, 16):
                        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                        input_embeddings = input_embeddings + codec_i_embedding

                    outputs = model.talker(
                        inputs_embeds=input_embeddings[:, :-1, :],
                        attention_mask=attention_mask[:, :-1],
                        labels=codec_0_labels[:, 1:],
                        output_hidden_states=True,
                    )

                    hidden_states = outputs.hidden_states[0][-1]
                    talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                    talker_codec_ids = codec_ids[codec_mask]
                    _, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                        talker_codec_ids, talker_hidden_states
                    )

                    main_loss = outputs.loss
                    sub_loss = sub_talker_loss
                    loss = main_loss + 0.3 * sub_loss
                    loss_value = float(loss.detach().float().cpu().item())
                    main_loss_value = float(main_loss.detach().float().cpu().item())
                    sub_loss_value = float(sub_loss.detach().float().cpu().item())
                    micro_token_count = int(codec_mask.sum().detach().cpu().item())

                    if not math.isfinite(loss_value) or not math.isfinite(main_loss_value) or not math.isfinite(sub_loss_value):
                        dump_dir = _dump_anomaly_batch(
                            anomaly_dir,
                            epoch=epoch,
                            step=effective_step,
                            global_step=global_step,
                            reason="non_finite_loss",
                            batch=batch,
                            exception_text="Detected non-finite loss before backward.",
                            latest_step_checkpoint=latest_step_checkpoint,
                        )
                        raise RuntimeError(f"Non-finite loss detected. Batch dump saved to {dump_dir}")

                    accelerator.backward(loss)
                    pending_loss_sum += loss_value
                    pending_main_loss_sum += main_loss_value
                    pending_sub_loss_sum += sub_loss_value
                    pending_micro_steps += 1
                    pending_token_count += micro_token_count

                    grad_norm_value = None
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        grad_norm_value = (
                            float(grad_norm.detach().float().cpu().item())
                            if torch.is_tensor(grad_norm)
                            else float(grad_norm)
                        )
                        if not math.isfinite(grad_norm_value):
                            dump_dir = _dump_anomaly_batch(
                                anomaly_dir,
                                epoch=epoch,
                                step=effective_step,
                                global_step=global_step,
                                reason="non_finite_grad_norm",
                                batch=batch,
                                exception_text="Detected non-finite grad_norm before optimizer step.",
                                latest_step_checkpoint=latest_step_checkpoint,
                            )
                            raise RuntimeError(f"Non-finite grad_norm detected. Batch dump saved to {dump_dir}")

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    current_lr = scheduler.get_last_lr()[0]
                    step_elapsed = max(time.perf_counter() - sync_window_start, 1e-6)
                    sync_window_start = time.perf_counter()

                    logged_loss_value = pending_loss_sum / max(1, pending_micro_steps)
                    logged_main_loss_value = pending_main_loss_sum / max(1, pending_micro_steps)
                    logged_sub_loss_value = pending_sub_loss_sum / max(1, pending_micro_steps)
                    token_count = pending_token_count
                    tokens_per_sec = token_count / step_elapsed

                    accelerator.log(
                        {
                            "train/loss": logged_loss_value,
                            "train/main_loss": logged_main_loss_value,
                            "train/sub_loss": logged_sub_loss_value,
                            "train/lr": current_lr,
                            "train/grad_norm": grad_norm_value if grad_norm_value is not None else 0.0,
                            "train/codec_tokens": token_count,
                            "train/tokens_per_sec": tokens_per_sec,
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

                    epoch_loss_sum += logged_loss_value
                    epoch_main_loss_sum += logged_main_loss_value
                    epoch_sub_loss_sum += logged_sub_loss_value
                    epoch_logged_steps += 1
                    epoch_token_count += token_count

                    row = {
                        "epoch": epoch,
                        "local_step": effective_step,
                        "global_step": global_step,
                        "loss": _safe_round(logged_loss_value, 8),
                        "main_loss": _safe_round(logged_main_loss_value, 8),
                        "sub_loss": _safe_round(logged_sub_loss_value, 8),
                        "lr": _safe_round(current_lr, 12),
                        "grad_norm": _safe_round(grad_norm_value, 8),
                        "codec_tokens": token_count,
                        "step_seconds": _safe_round(step_elapsed, 4),
                        "tokens_per_sec": _safe_round(tokens_per_sec, 4),
                    }
                    if accelerator.is_main_process:
                        append_metrics_csv(step_csv, row)

                    tracker.mark_progress(
                        next_epoch=epoch,
                        next_batch_in_epoch=effective_step + 1,
                        global_step=global_step,
                        target_speaker_embedding=target_speaker_embedding,
                        last_step_checkpoint=latest_step_checkpoint,
                        last_reason="step_completed",
                    )

                    if args.save_training_state_steps > 0 and global_step % args.save_training_state_steps == 0:
                        ckpt_dir = _save_training_state_checkpoint(
                            accelerator,
                            tracker,
                            training_state_dir,
                            args.keep_last_training_states,
                            reason="step_interval",
                        )
                        latest_step_checkpoint = str(ckpt_dir)
                        tracker.last_step_checkpoint = latest_step_checkpoint
                        if accelerator.is_main_process:
                            _write_json(
                                training_state_dir / "latest.json",
                                {
                                    **tracker.state_dict(),
                                    "checkpoint_dir": latest_step_checkpoint,
                                    "reason": "step_interval",
                                },
                            )

                    if global_step % args.log_steps == 0:
                        grad_norm_text = "n/a" if grad_norm_value is None else f"{grad_norm_value:.4f}"
                        accelerator.print(
                            f"Epoch {epoch} | Step {effective_step} | GlobalStep {global_step} | "
                            f"Loss {logged_loss_value:.4f} | Main {logged_main_loss_value:.4f} | "
                            f"Sub {logged_sub_loss_value:.4f} | GradNorm {grad_norm_text} | LR {current_lr:.2e}"
                        )

                    pending_loss_sum = 0.0
                    pending_main_loss_sum = 0.0
                    pending_sub_loss_sum = 0.0
                    pending_micro_steps = 0
                    pending_token_count = 0
            except Exception as exc:
                tb = traceback.format_exc()
                dump_dir = _dump_anomaly_batch(
                    anomaly_dir,
                    epoch=epoch,
                    step=effective_step,
                    global_step=global_step,
                    reason=type(exc).__name__,
                    batch=batch,
                    exception_text=tb,
                    latest_step_checkpoint=latest_step_checkpoint,
                )
                if accelerator.is_main_process:
                    _write_json(
                        anomaly_dir / "latest_failure.json",
                        {
                            "epoch": epoch,
                            "step": effective_step,
                            "global_step": global_step,
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc),
                            "dump_dir": str(dump_dir),
                            "recommended_resume_checkpoint": latest_step_checkpoint,
                        },
                    )
                raise

        elapsed_sec = time.perf_counter() - epoch_wall_start
        avg_epoch_loss = epoch_loss_sum / max(1, epoch_logged_steps)
        avg_epoch_main_loss = epoch_main_loss_sum / max(1, epoch_logged_steps)
        avg_epoch_sub_loss = epoch_sub_loss_sum / max(1, epoch_logged_steps)
        epoch_tokens_per_sec = epoch_token_count / elapsed_sec if elapsed_sec > 0 else float("nan")

        accelerator.log(
            {
                "epoch/loss": avg_epoch_loss,
                "epoch/main_loss": avg_epoch_main_loss,
                "epoch/sub_loss": avg_epoch_sub_loss,
                "epoch/tokens_per_sec": epoch_tokens_per_sec if not math.isnan(epoch_tokens_per_sec) else 0.0,
            },
            step=global_step,
        )

        summary = {
            "epoch": epoch,
            "avg_epoch_loss": avg_epoch_loss,
            "avg_epoch_main_loss": avg_epoch_main_loss,
            "avg_epoch_sub_loss": avg_epoch_sub_loss,
            "epoch_tokens": epoch_token_count,
            "epoch_seconds": None if math.isnan(elapsed_sec) else elapsed_sec,
            "epoch_tokens_per_sec": None if math.isnan(epoch_tokens_per_sec) else epoch_tokens_per_sec,
            "global_step_end": global_step,
            "fixed_eval_jsonl": "" if fixed_eval_path is None else str(fixed_eval_path),
            "last_step_checkpoint": latest_step_checkpoint,
        }

        tracker.mark_progress(
            next_epoch=epoch + 1,
            next_batch_in_epoch=0,
            global_step=global_step,
            target_speaker_embedding=target_speaker_embedding,
            last_step_checkpoint=latest_step_checkpoint,
            last_reason="epoch_completed",
        )
        epoch_state_ckpt = _save_training_state_checkpoint(
            accelerator,
            tracker,
            training_state_dir,
            args.keep_last_training_states,
            reason="epoch_completed",
        )
        latest_step_checkpoint = str(epoch_state_ckpt)
        tracker.last_step_checkpoint = latest_step_checkpoint
        summary["last_step_checkpoint"] = latest_step_checkpoint

        if accelerator.is_main_process:
            append_metrics_csv(
                epoch_csv,
                {
                    "epoch": epoch,
                    "epoch_loss": _safe_round(avg_epoch_loss, 8),
                    "epoch_main_loss": _safe_round(avg_epoch_main_loss, 8),
                    "epoch_sub_loss": _safe_round(avg_epoch_sub_loss, 8),
                    "epoch_tokens": epoch_token_count,
                    "epoch_seconds": _safe_round(elapsed_sec, 4),
                    "epoch_tokens_per_sec": _safe_round(epoch_tokens_per_sec, 4),
                    "global_step_end": global_step,
                },
            )
            if args.save_plots_every_epoch:
                plotter.plot_step_metrics(step_csv)
                plotter.plot_epoch_metrics(epoch_csv)

            ckpt_dir = _export_inference_checkpoint(
                accelerator=accelerator,
                model=model,
                model_path=model_path,
                output_model_path=output_dir,
                epoch=epoch,
                speaker_name=args.speaker_name,
                target_embedding=target_speaker_embedding,
                summary=summary,
            )

            if not args.skip_fixed_eval_generation and fixed_eval_rows:
                _render_fixed_eval_samples(
                    accelerator=accelerator,
                    model=model,
                    processor=qwen3tts.processor,
                    speaker_name=args.speaker_name,
                    target_embedding=target_speaker_embedding,
                    fixed_eval_rows=fixed_eval_rows,
                    output_dir=fixed_eval_audio_dir / ckpt_dir.name,
                    max_new_tokens=args.fixed_eval_max_new_tokens,
                )

        tracker.next_batch_in_epoch = 0
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        plotter.plot_step_metrics(step_csv)
        plotter.plot_epoch_metrics(epoch_csv)

    accelerator.end_training()


if __name__ == "__main__":
    train()
