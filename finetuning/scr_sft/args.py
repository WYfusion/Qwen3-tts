# coding=utf-8

from __future__ import annotations

import argparse

from .config import CheckpointConfig, EvalConfig, LoggingConfig, RunPaths, SFTConfig, TrainConfig

RECIPE_CHOICES = ["staged_benchmark_aligned_sft", "staged_stable_sft", "legacy_full_sft"]


def _resolve_recipe_aware_defaults(args: argparse.Namespace) -> dict[str, int]:
    recipe = str(args.training_recipe)
    recipe_defaults = {
        "fixed_eval_num_samples": 24 if recipe == "staged_benchmark_aligned_sft" else 4,
        "early_stop_patience": 2 if recipe == "staged_benchmark_aligned_sft" else 1,
        "stage1_epochs": 1,
        "stage2_epochs": 2 if recipe == "staged_benchmark_aligned_sft" else 3,
        "stage3_epochs": 2 if recipe == "staged_benchmark_aligned_sft" else 0,
        "benchmark_eval_num_samples": 48 if recipe == "staged_benchmark_aligned_sft" else 0,
        "benchmark_eval_every_epochs": 1,
    }
    resolved = {}
    for key, default_value in recipe_defaults.items():
        value = getattr(args, key)
        resolved[key] = int(default_value if value is None else value)
    return resolved


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--fixed_eval_source_jsonl", type=str, default=None)
    parser.add_argument("--fixed_eval_num_samples", type=int, default=None)
    parser.add_argument("--fixed_eval_length_mode", type=str, choices=["dynamic", "fixed"], default="dynamic")
    parser.add_argument("--fixed_eval_length_multiplier", type=float, default=2.0)
    parser.add_argument("--fixed_eval_max_new_tokens", type=int, default=256)
    parser.add_argument("--fixed_eval_language", type=str, default="Chinese")
    parser.add_argument("--fixed_eval_do_sample", action="store_true")
    parser.add_argument("--fixed_eval_duration_ratio_warn", type=float, default=1.5)
    parser.add_argument("--training_recipe", type=str, choices=RECIPE_CHOICES, default="staged_benchmark_aligned_sft")
    parser.add_argument("--stage1_epochs", type=int, default=None)
    parser.add_argument("--stage2_epochs", type=int, default=None)
    parser.add_argument("--stage3_epochs", type=int, default=None)
    parser.add_argument("--main_kd_weight", type=float, default=0.1)
    parser.add_argument("--sub_kd_weight", type=float, default=0.05)
    parser.add_argument("--kd_temperature", type=float, default=2.0)
    parser.add_argument("--speaker_init_num_samples", type=int, default=16)
    parser.add_argument("--enable_free_run_eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--free_run_eval_max_new_tokens", type=int, default=8192)
    parser.add_argument("--benchmark_eval_jsonl", type=str, default=None)
    parser.add_argument("--benchmark_eval_num_samples", type=int, default=None)
    parser.add_argument("--benchmark_eval_every_epochs", type=int, default=None)
    parser.add_argument("--seed_tts_eval_root", type=str, default="./seed-tts-eval")
    parser.add_argument("--seed_tts_eval_python", type=str, default="python3")
    parser.add_argument("--benchmark_eval_device", type=str, default="cuda:0")
    parser.add_argument("--sim_finetune_checkpoint", type=str, default="./seed-tts-eval/weight/wavlm_large_finetune.pth")
    parser.add_argument("--peak_warn_threshold", type=float, default=0.99)
    parser.add_argument("--clipped_frac_warn_threshold", type=float, default=1e-6)
    parser.add_argument("--hf_noise_warn_threshold", type=float, default=0.12)
    parser.add_argument("--voiced_f0_delta_warn_threshold", type=float, default=180.0)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--audio_qc_report_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--log_with", type=str, default="tensorboard,wandb")
    parser.add_argument("--wandb_project", type=str, default="qwen3-tts-finetune")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default="")
    parser.add_argument("--wandb_job_type", type=str, default="train")
    parser.add_argument("--wandb_resume", type=str, choices=["never", "allow", "must"], default="allow")
    parser.add_argument("--skip_fixed_eval_generation", action="store_true")
    return parser


def namespace_to_config(args: argparse.Namespace) -> SFTConfig:
    resolved_defaults = _resolve_recipe_aware_defaults(args)
    train = TrainConfig(
        init_model_path=args.init_model_path,
        output_model_path=args.output_model_path,
        train_jsonl=args.train_jsonl,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        speaker_name=args.speaker_name,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        log_steps=args.log_steps,
        save_plots_every_epoch=args.save_plots_every_epoch,
        disable_flash_attn=args.disable_flash_attn,
        seed=args.seed,
        training_recipe=args.training_recipe,
        stage1_epochs=resolved_defaults["stage1_epochs"],
        stage2_epochs=resolved_defaults["stage2_epochs"],
        stage3_epochs=resolved_defaults["stage3_epochs"],
        main_kd_weight=args.main_kd_weight,
        sub_kd_weight=args.sub_kd_weight,
        kd_temperature=args.kd_temperature,
        speaker_init_num_samples=args.speaker_init_num_samples,
        audio_qc_report_only=args.audio_qc_report_only,
        dry_run=args.dry_run,
    )
    eval_config = EvalConfig(
        fixed_eval_jsonl=args.fixed_eval_jsonl,
        fixed_eval_source_jsonl=args.fixed_eval_source_jsonl,
        fixed_eval_num_samples=resolved_defaults["fixed_eval_num_samples"],
        fixed_eval_length_mode=args.fixed_eval_length_mode,
        fixed_eval_length_multiplier=args.fixed_eval_length_multiplier,
        fixed_eval_max_new_tokens=args.fixed_eval_max_new_tokens,
        fixed_eval_language=args.fixed_eval_language,
        fixed_eval_do_sample=args.fixed_eval_do_sample,
        fixed_eval_duration_ratio_warn=args.fixed_eval_duration_ratio_warn,
        enable_free_run_eval=args.enable_free_run_eval,
        free_run_eval_max_new_tokens=args.free_run_eval_max_new_tokens,
        benchmark_eval_jsonl=args.benchmark_eval_jsonl,
        benchmark_eval_num_samples=resolved_defaults["benchmark_eval_num_samples"],
        benchmark_eval_every_epochs=resolved_defaults["benchmark_eval_every_epochs"],
        seed_tts_eval_root=args.seed_tts_eval_root,
        seed_tts_eval_python=args.seed_tts_eval_python,
        benchmark_eval_device=args.benchmark_eval_device,
        sim_finetune_checkpoint=args.sim_finetune_checkpoint,
        peak_warn_threshold=args.peak_warn_threshold,
        clipped_frac_warn_threshold=args.clipped_frac_warn_threshold,
        hf_noise_warn_threshold=args.hf_noise_warn_threshold,
        voiced_f0_delta_warn_threshold=args.voiced_f0_delta_warn_threshold,
        skip_fixed_eval_generation=args.skip_fixed_eval_generation,
    )
    logging = LoggingConfig(
        log_with_raw=args.log_with,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
        wandb_job_type=args.wandb_job_type,
        wandb_resume=args.wandb_resume,
    )
    checkpoint = CheckpointConfig(
        resume_from_training_state=args.resume_from_training_state,
        save_training_state_steps=args.save_training_state_steps,
        keep_last_training_states=args.keep_last_training_states,
        early_stop_patience=resolved_defaults["early_stop_patience"],
    )
    paths = RunPaths.from_output_root(args.output_model_path)
    return SFTConfig(train=train, eval=eval_config, logging=logging, checkpoint=checkpoint, paths=paths)
