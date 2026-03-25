# coding=utf-8

from __future__ import annotations

import math
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.data_loader import skip_first_batches
from transformers import AutoConfig, get_cosine_schedule_with_warmup

from .audio_qc import build_audio_qc_report
from .data import (
    TTSDataset,
    build_epoch_dataloader,
    dataset_stats,
    infer_dataset_name,
    infer_fixed_eval_source_path,
    prepare_fixed_eval_set,
)
from .eval_audio import render_eval_samples
from .export import export_inference_checkpoint, write_best_checkpoint_record
from .io_utils import append_metrics_csv, json_ready, read_jsonl, safe_round, write_json
from .losses import compose_total_loss, kl_div_with_temperature, masked_main_logits
from .model_ops import forward_talker_with_sub_loss, load_qwen3tts_with_attn_fallback, maybe_init_custom_speaker_row
from .plots import MetricsPlotter
from .recipes import apply_training_stage, build_optimizer, effective_num_epochs, register_speaker_row_gradient_mask
from .state import (
    TrainingStateTracker,
    dump_anomaly_batch,
    load_training_state_metadata,
    resolve_resume_checkpoint,
    save_training_state_checkpoint,
)
from .tracking import default_wandb_group, default_wandb_run_name, init_trackers, parse_log_with


class SFTTrainer:
    def __init__(self, config):
        self.config = config
        self.train_cfg = config.train
        self.eval_cfg = config.eval
        self.logging_cfg = config.logging
        self.checkpoint_cfg = config.checkpoint
        self.paths = config.paths

        self.plotter = MetricsPlotter(self.paths.plots_dir)
        self.parsed_log_with = parse_log_with(self.logging_cfg.log_with_raw)
        self.effective_epochs = effective_num_epochs(self.train_cfg)
        self.dataset_name = infer_dataset_name(self.train_cfg.train_jsonl)
        self.resume_checkpoint = resolve_resume_checkpoint(
            self.checkpoint_cfg.resume_from_training_state,
            self.paths.training_state_dir,
        )
        self.resume_metadata = load_training_state_metadata(self.resume_checkpoint)
        self.wandb_run_id = str(self.resume_metadata.get("wandb_run_id", ""))
        resumed_name = str(self.resume_metadata.get("wandb_run_name", ""))
        self.wandb_run_name = self.logging_cfg.wandb_run_name or resumed_name or default_wandb_run_name(
            self.train_cfg,
            self.effective_epochs,
        )
        self.wandb_group = self.logging_cfg.wandb_group or default_wandb_group(self.train_cfg, self.dataset_name)

        self.accelerator = None
        self.qwen3tts = None
        self.processor = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.teacher_model = None
        self.speaker_row_hook = None
        self.tracker = None
        self.grouped_param_names = {}

        self.train_data = []
        self.train_stats = {}
        self.fixed_eval_rows = []
        self.fixed_eval_path = None
        self.fixed_eval_source_path = None
        self.ref_audio_used = ""

        self.global_step = 0
        self.latest_step_checkpoint = ""
        self.start_epoch = 0
        self.should_exit = False

    def setup(self):
        torch.manual_seed(self.train_cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.train_cfg.seed)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.train_cfg.gradient_accumulation_steps,
            mixed_precision="bf16",
            log_with=self.parsed_log_with,
            project_dir=str(self.paths.tb_dir),
        )

        if self.effective_epochs <= 0:
            self.accelerator.print("No epochs scheduled. Exiting without training.")
            self.should_exit = True
            return

        attn_impl = None if self.train_cfg.disable_flash_attn else "flash_attention_2"
        self.qwen3tts = load_qwen3tts_with_attn_fallback(
            self.train_cfg.init_model_path,
            attn_impl=attn_impl,
            accelerator=self.accelerator,
        )
        self.processor = self.qwen3tts.processor
        config = AutoConfig.from_pretrained(self.train_cfg.init_model_path)
        self.train_data = read_jsonl(self.train_cfg.train_jsonl)
        self.train_stats = dataset_stats(self.train_data)
        self.dataset = TTSDataset(self.train_data, self.processor, config)
        self.ref_audio_used = maybe_init_custom_speaker_row(
            self.qwen3tts.model,
            self.dataset,
            self.train_data,
            self.accelerator,
        )

        if self.accelerator.is_main_process:
            qc_result = build_audio_qc_report(
                self.train_data,
                output_dir=self.paths.qc_report_dir,
                peak_warn=self.eval_cfg.peak_warn_threshold,
                clip_warn=self.eval_cfg.clipped_frac_warn_threshold,
            )
            self.accelerator.print(
                f"Audio QC report written to {self.paths.qc_report_dir} | "
                f"peak_warn={qc_result['summary']['num_peak_warn']} | "
                f"clip_warn={qc_result['summary']['num_clip_warn']}"
            )
        self.accelerator.wait_for_everyone()
        if self.train_cfg.audio_qc_report_only:
            self.accelerator.print("--audio_qc_report_only enabled; exiting after QC report generation.")
            self.should_exit = True
            return

        if self.eval_cfg.fixed_eval_num_samples > 0 and self.accelerator.is_main_process:
            self.fixed_eval_rows, self.fixed_eval_path, self.fixed_eval_source_path = prepare_fixed_eval_set(
                train_jsonl=self.train_cfg.train_jsonl,
                fixed_eval_jsonl=self.eval_cfg.fixed_eval_jsonl,
                fixed_eval_source_jsonl=self.eval_cfg.fixed_eval_source_jsonl,
                fixed_eval_num_samples=self.eval_cfg.fixed_eval_num_samples,
                fixed_eval_language=self.eval_cfg.fixed_eval_language,
                speaker_name=self.train_cfg.speaker_name,
                speech_tokenizer=self.qwen3tts.model.speech_tokenizer,
                logs_dir=self.paths.logs_dir,
            )
        self.accelerator.wait_for_everyone()
        if self.eval_cfg.fixed_eval_num_samples > 0:
            self.fixed_eval_path = (
                Path(self.eval_cfg.fixed_eval_jsonl)
                if self.eval_cfg.fixed_eval_jsonl
                else self.paths.logs_dir / "fixed_eval_set.jsonl"
            )
            if self.fixed_eval_path.exists():
                self.fixed_eval_rows = read_jsonl(self.fixed_eval_path)
            self.fixed_eval_source_path = (
                Path(self.eval_cfg.fixed_eval_source_jsonl)
                if self.eval_cfg.fixed_eval_source_jsonl
                else infer_fixed_eval_source_path(self.train_cfg.train_jsonl)
            )

        tracker_config = json_ready(asdict(self.config))
        tracker_config.update(
            {
                "dataset_name": self.dataset_name,
                "effective_num_epochs": self.effective_epochs,
                "fixed_eval_jsonl_resolved": "" if self.fixed_eval_path is None else str(self.fixed_eval_path),
                "fixed_eval_source_jsonl_resolved": "" if self.fixed_eval_source_path is None else str(self.fixed_eval_source_path),
                "ref_audio_used": self.ref_audio_used,
                **self.train_stats,
            }
        )

        if not self.train_cfg.dry_run and self.accelerator.is_main_process:
            self.wandb_run_id, self.wandb_run_name = init_trackers(
                self.accelerator,
                logging_config=self.logging_cfg,
                tracker_config=tracker_config,
                wandb_run_id=self.wandb_run_id,
                wandb_run_name=self.wandb_run_name,
                wandb_group=self.wandb_group,
            )

        optimizer, _, self.grouped_param_names = build_optimizer(self.qwen3tts.model, self.train_cfg)
        steps_per_epoch = math.ceil(len(self.dataset) / self.train_cfg.batch_size / self.train_cfg.gradient_accumulation_steps)
        total_training_steps = max(1, self.effective_epochs * steps_per_epoch)
        warmup_steps = int(total_training_steps * self.train_cfg.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(self.qwen3tts.model, optimizer, scheduler)
        if self.train_cfg.training_recipe == "staged_stable_sft":
            self.speaker_row_hook = register_speaker_row_gradient_mask(
                self.accelerator.unwrap_model(self.model),
                custom_speaker_id=3000,
            )

        if self.train_cfg.main_kd_weight > 0.0 or self.train_cfg.sub_kd_weight > 0.0:
            self.teacher_model = load_qwen3tts_with_attn_fallback(
                self.train_cfg.init_model_path,
                attn_impl=attn_impl,
                accelerator=self.accelerator,
            ).model
            self.teacher_model.eval()
            self.teacher_model.to(self.accelerator.device)
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.accelerator.print("Loaded frozen base teacher for KD.")

        self.tracker = TrainingStateTracker()
        self.accelerator.register_for_checkpointing(self.tracker)
        if self.resume_checkpoint is not None:
            self.accelerator.load_state(str(self.resume_checkpoint))
            self.accelerator.print(
                f"Resumed from {self.resume_checkpoint} | next_epoch={self.tracker.next_epoch} "
                f"| next_batch_in_epoch={self.tracker.next_batch_in_epoch} | global_step={self.tracker.global_step}"
            )
            if not self.wandb_run_id and self.tracker.wandb_run_id:
                self.wandb_run_id = self.tracker.wandb_run_id
            if not self.wandb_run_name and self.tracker.wandb_run_name:
                self.wandb_run_name = self.tracker.wandb_run_name

        self.tracker.wandb_run_id = self.wandb_run_id
        self.tracker.wandb_run_name = self.wandb_run_name
        self.global_step = self.tracker.global_step
        self.latest_step_checkpoint = self.tracker.last_step_checkpoint
        self.start_epoch = self.tracker.next_epoch
        self.accelerator.print(
            f"Training recipe: {self.train_cfg.training_recipe} | total_epochs={self.effective_epochs}"
        )

    def run(self):
        self.setup()
        if self.should_exit:
            self._finalize_run()
            return

        if self.train_cfg.dry_run:
            self._run_dry_run()
            self._finalize_run()
            return

        self.model.train()
        for epoch in range(self.start_epoch, self.effective_epochs):
            should_stop = self._train_epoch(epoch)
            if should_stop:
                break
        self._finalize_run()

    def _run_dry_run(self):
        apply_training_stage(
            self.accelerator.unwrap_model(self.model),
            self.optimizer,
            self.train_cfg,
            0,
            self.accelerator,
            self.grouped_param_names,
        )
        if self.effective_epochs > 1:
            apply_training_stage(
                self.accelerator.unwrap_model(self.model),
                self.optimizer,
                self.train_cfg,
                min(self.effective_epochs - 1, max(self.train_cfg.stage1_epochs, 0)),
                self.accelerator,
                self.grouped_param_names,
            )
        self.accelerator.print("--dry_run enabled; exiting before optimizer steps.")

    def _train_epoch(self, epoch: int) -> bool:
        current_stage = apply_training_stage(
            self.accelerator.unwrap_model(self.model),
            self.optimizer,
            self.train_cfg,
            epoch,
            self.accelerator,
            self.grouped_param_names,
        )
        raw_epoch_dataloader = build_epoch_dataloader(
            self.dataset,
            self.train_cfg.batch_size,
            self.dataset.collate_fn,
            epoch,
            self.train_cfg.seed,
        )
        train_dataloader = self.accelerator.prepare_data_loader(raw_epoch_dataloader)
        resume_batches = self.tracker.next_batch_in_epoch if epoch == self.start_epoch else 0
        if resume_batches > 0:
            train_dataloader = skip_first_batches(train_dataloader, resume_batches)

        epoch_loss_sum = 0.0
        epoch_main_loss_sum = 0.0
        epoch_sub_loss_sum = 0.0
        epoch_main_kd_loss_sum = 0.0
        epoch_sub_kd_loss_sum = 0.0
        epoch_logged_steps = 0
        epoch_token_count = 0
        pending_loss_sum = 0.0
        pending_main_loss_sum = 0.0
        pending_sub_loss_sum = 0.0
        pending_main_kd_sum = 0.0
        pending_sub_kd_sum = 0.0
        pending_micro_steps = 0
        pending_token_count = 0
        sync_window_start = time.perf_counter()
        epoch_wall_start = time.perf_counter()
        effective_step = resume_batches - 1

        for batch in train_dataloader:
            effective_step += 1
            try:
                with self.accelerator.accumulate(self.model):
                    outputs, sub_logits, sub_talker_loss = forward_talker_with_sub_loss(self.model, batch)
                    main_loss = outputs.loss
                    sub_loss = sub_talker_loss
                    main_kd_loss = main_loss.new_tensor(0.0)
                    sub_kd_loss = main_loss.new_tensor(0.0)
                    if self.teacher_model is not None and (
                        self.train_cfg.main_kd_weight > 0.0 or self.train_cfg.sub_kd_weight > 0.0
                    ):
                        with torch.no_grad():
                            teacher_outputs, teacher_sub_logits, _ = forward_talker_with_sub_loss(self.teacher_model, batch)
                        student_main_logits = masked_main_logits(outputs.logits, batch["codec_0_labels"][:, 1:])
                        teacher_main_logits = masked_main_logits(teacher_outputs.logits, batch["codec_0_labels"][:, 1:])
                        if self.train_cfg.main_kd_weight > 0.0:
                            main_kd_loss = kl_div_with_temperature(
                                student_main_logits,
                                teacher_main_logits,
                                self.train_cfg.kd_temperature,
                            )
                        if self.train_cfg.sub_kd_weight > 0.0:
                            sub_kd_loss = kl_div_with_temperature(
                                sub_logits,
                                teacher_sub_logits,
                                self.train_cfg.kd_temperature,
                            )

                    loss = compose_total_loss(
                        main_loss,
                        sub_loss,
                        main_kd_loss,
                        sub_kd_loss,
                        main_kd_weight=self.train_cfg.main_kd_weight,
                        sub_kd_weight=self.train_cfg.sub_kd_weight,
                    )
                    values = {
                        "loss": float(loss.detach().float().cpu().item()),
                        "main_loss": float(main_loss.detach().float().cpu().item()),
                        "sub_loss": float(sub_loss.detach().float().cpu().item()),
                        "main_kd_loss": float(main_kd_loss.detach().float().cpu().item()),
                        "sub_kd_loss": float(sub_kd_loss.detach().float().cpu().item()),
                    }
                    if not all(math.isfinite(v) for v in values.values()):
                        dump_dir = dump_anomaly_batch(
                            self.paths.anomaly_dir,
                            epoch=epoch,
                            step=effective_step,
                            global_step=self.global_step,
                            reason="non_finite_loss",
                            batch=batch,
                            exception_text="Detected non-finite loss before backward.",
                            latest_step_checkpoint=self.latest_step_checkpoint,
                        )
                        raise RuntimeError(f"Non-finite loss detected. Batch dump saved to {dump_dir}")

                    self.accelerator.backward(loss)
                    pending_loss_sum += values["loss"]
                    pending_main_loss_sum += values["main_loss"]
                    pending_sub_loss_sum += values["sub_loss"]
                    pending_main_kd_sum += values["main_kd_loss"]
                    pending_sub_kd_sum += values["sub_kd_loss"]
                    pending_micro_steps += 1
                    pending_token_count += int(batch["codec_mask"].sum().detach().cpu().item())

                    grad_norm_value = None
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.train_cfg.max_grad_norm)
                        grad_norm_value = (
                            float(grad_norm.detach().float().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
                        )
                        if not math.isfinite(grad_norm_value):
                            dump_dir = dump_anomaly_batch(
                                self.paths.anomaly_dir,
                                epoch=epoch,
                                step=effective_step,
                                global_step=self.global_step,
                                reason="non_finite_grad_norm",
                                batch=batch,
                                exception_text="Detected non-finite grad_norm before optimizer step.",
                                latest_step_checkpoint=self.latest_step_checkpoint,
                            )
                            raise RuntimeError(f"Non-finite grad_norm detected. Batch dump saved to {dump_dir}")

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    current_lr = max(float(group["lr"]) for group in self.optimizer.param_groups)
                    step_elapsed = max(time.perf_counter() - sync_window_start, 1e-6)
                    sync_window_start = time.perf_counter()
                    logged = {
                        "loss": pending_loss_sum / max(1, pending_micro_steps),
                        "main_loss": pending_main_loss_sum / max(1, pending_micro_steps),
                        "sub_loss": pending_sub_loss_sum / max(1, pending_micro_steps),
                        "main_kd_loss": pending_main_kd_sum / max(1, pending_micro_steps),
                        "sub_kd_loss": pending_sub_kd_sum / max(1, pending_micro_steps),
                    }
                    token_count = pending_token_count
                    tokens_per_sec = token_count / step_elapsed
                    self.accelerator.log(
                        {
                            "train/loss": logged["loss"],
                            "train/main_loss": logged["main_loss"],
                            "train/sub_loss": logged["sub_loss"],
                            "train/main_kd_loss": logged["main_kd_loss"],
                            "train/sub_kd_loss": logged["sub_kd_loss"],
                            "train/lr": current_lr,
                            "train/grad_norm": grad_norm_value if grad_norm_value is not None else 0.0,
                            "train/codec_tokens": token_count,
                            "train/tokens_per_sec": tokens_per_sec,
                            "train/epoch": epoch,
                        },
                        step=self.global_step,
                    )
                    epoch_loss_sum += logged["loss"]
                    epoch_main_loss_sum += logged["main_loss"]
                    epoch_sub_loss_sum += logged["sub_loss"]
                    epoch_main_kd_loss_sum += logged["main_kd_loss"]
                    epoch_sub_kd_loss_sum += logged["sub_kd_loss"]
                    epoch_logged_steps += 1
                    epoch_token_count += token_count
                    if self.accelerator.is_main_process:
                        append_metrics_csv(
                            self.paths.step_csv,
                            {
                                "epoch": epoch,
                                "local_step": effective_step,
                                "global_step": self.global_step,
                                "loss": safe_round(logged["loss"], 8),
                                "main_loss": safe_round(logged["main_loss"], 8),
                                "sub_loss": safe_round(logged["sub_loss"], 8),
                                "main_kd_loss": safe_round(logged["main_kd_loss"], 8),
                                "sub_kd_loss": safe_round(logged["sub_kd_loss"], 8),
                                "lr": safe_round(current_lr, 12),
                                "grad_norm": safe_round(grad_norm_value, 8),
                                "codec_tokens": token_count,
                                "step_seconds": safe_round(step_elapsed, 4),
                                "tokens_per_sec": safe_round(tokens_per_sec, 4),
                            },
                        )

                    self.tracker.mark_progress(
                        next_epoch=epoch,
                        next_batch_in_epoch=effective_step + 1,
                        global_step=self.global_step,
                        last_step_checkpoint=self.latest_step_checkpoint,
                        last_reason="step_completed",
                        wandb_run_id=self.wandb_run_id,
                        wandb_run_name=self.wandb_run_name,
                    )
                    if (
                        self.checkpoint_cfg.save_training_state_steps > 0
                        and self.global_step % self.checkpoint_cfg.save_training_state_steps == 0
                    ):
                        ckpt_dir = save_training_state_checkpoint(
                            self.accelerator,
                            self.tracker,
                            self.paths.training_state_dir,
                            self.checkpoint_cfg.keep_last_training_states,
                            reason="step_interval",
                        )
                        self.latest_step_checkpoint = str(ckpt_dir)

                    if self.global_step % self.train_cfg.log_steps == 0:
                        grad_norm_text = "n/a" if grad_norm_value is None else f"{grad_norm_value:.4f}"
                        self.accelerator.print(
                            f"Epoch {epoch} [{current_stage}] | Step {effective_step} | GlobalStep {self.global_step} | "
                            f"Loss {logged['loss']:.4f} | Main {logged['main_loss']:.4f} | "
                            f"Sub {logged['sub_loss']:.4f} | MainKD {logged['main_kd_loss']:.4f} | "
                            f"SubKD {logged['sub_kd_loss']:.4f} | GradNorm {grad_norm_text} | LR {current_lr:.2e}"
                        )

                    pending_loss_sum = 0.0
                    pending_main_loss_sum = 0.0
                    pending_sub_loss_sum = 0.0
                    pending_main_kd_sum = 0.0
                    pending_sub_kd_sum = 0.0
                    pending_micro_steps = 0
                    pending_token_count = 0
            except Exception as exc:
                tb = traceback.format_exc()
                dump_dir = dump_anomaly_batch(
                    self.paths.anomaly_dir,
                    epoch=epoch,
                    step=effective_step,
                    global_step=self.global_step,
                    reason=type(exc).__name__,
                    batch=batch,
                    exception_text=tb,
                    latest_step_checkpoint=self.latest_step_checkpoint,
                )
                if self.accelerator.is_main_process:
                    write_json(
                        self.paths.anomaly_dir / "latest_failure.json",
                        {
                            "epoch": epoch,
                            "step": effective_step,
                            "global_step": self.global_step,
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc),
                            "dump_dir": str(dump_dir),
                            "recommended_resume_checkpoint": self.latest_step_checkpoint,
                        },
                    )
                raise

        elapsed_sec = time.perf_counter() - epoch_wall_start
        avg_epoch_loss = epoch_loss_sum / max(1, epoch_logged_steps)
        avg_epoch_main_loss = epoch_main_loss_sum / max(1, epoch_logged_steps)
        avg_epoch_sub_loss = epoch_sub_loss_sum / max(1, epoch_logged_steps)
        avg_epoch_main_kd_loss = epoch_main_kd_loss_sum / max(1, epoch_logged_steps)
        avg_epoch_sub_kd_loss = epoch_sub_kd_loss_sum / max(1, epoch_logged_steps)
        epoch_tokens_per_sec = epoch_token_count / elapsed_sec if elapsed_sec > 0 else float("nan")
        self.accelerator.log(
            {
                "epoch/loss": avg_epoch_loss,
                "epoch/main_loss": avg_epoch_main_loss,
                "epoch/sub_loss": avg_epoch_sub_loss,
                "epoch/main_kd_loss": avg_epoch_main_kd_loss,
                "epoch/sub_kd_loss": avg_epoch_sub_kd_loss,
                "epoch/tokens_per_sec": epoch_tokens_per_sec if not math.isnan(epoch_tokens_per_sec) else 0.0,
            },
            step=self.global_step,
        )

        summary = {
            "epoch": epoch,
            "stage_name": current_stage,
            "avg_epoch_loss": avg_epoch_loss,
            "avg_epoch_main_loss": avg_epoch_main_loss,
            "avg_epoch_sub_loss": avg_epoch_sub_loss,
            "avg_epoch_main_kd_loss": avg_epoch_main_kd_loss,
            "avg_epoch_sub_kd_loss": avg_epoch_sub_kd_loss,
            "epoch_tokens": epoch_token_count,
            "epoch_seconds": None if math.isnan(elapsed_sec) else elapsed_sec,
            "epoch_tokens_per_sec": None if math.isnan(epoch_tokens_per_sec) else epoch_tokens_per_sec,
            "global_step_end": self.global_step,
            "fixed_eval_jsonl": "" if self.fixed_eval_path is None else str(self.fixed_eval_path),
            "last_step_checkpoint": self.latest_step_checkpoint,
            "wandb_run_id": self.wandb_run_id,
            "wandb_run_name": self.wandb_run_name,
            "training_recipe": self.train_cfg.training_recipe,
        }
        return self._finalize_epoch(epoch=epoch, current_stage=current_stage, summary=summary)

    def _finalize_epoch(self, *, epoch: int, current_stage: str, summary: dict) -> bool:
        self.tracker.mark_progress(
            next_epoch=epoch + 1,
            next_batch_in_epoch=0,
            global_step=self.global_step,
            last_step_checkpoint=self.latest_step_checkpoint,
            last_reason="epoch_completed",
            wandb_run_id=self.wandb_run_id,
            wandb_run_name=self.wandb_run_name,
        )
        epoch_state_ckpt = save_training_state_checkpoint(
            self.accelerator,
            self.tracker,
            self.paths.training_state_dir,
            self.checkpoint_cfg.keep_last_training_states,
            reason="epoch_completed",
        )
        self.latest_step_checkpoint = str(epoch_state_ckpt)
        summary["last_step_checkpoint"] = self.latest_step_checkpoint

        fixed_eval_result = None
        free_run_eval_result = None
        should_stop = False
        early_stop_reason = ""

        if self.accelerator.is_main_process:
            ckpt_dir = export_inference_checkpoint(
                accelerator=self.accelerator,
                model=self.model,
                model_path=self.train_cfg.init_model_path,
                output_model_path=self.paths.output_dir,
                epoch=epoch,
                speaker_name=self.train_cfg.speaker_name,
                summary=summary,
            )
            if not self.eval_cfg.skip_fixed_eval_generation and self.fixed_eval_rows:
                fixed_eval_result = render_eval_samples(
                    accelerator=self.accelerator,
                    model=self.model,
                    processor=self.processor,
                    speaker_name=self.train_cfg.speaker_name,
                    eval_rows=self.fixed_eval_rows,
                    output_dir=self.paths.fixed_eval_audio_dir / ckpt_dir.name,
                    eval_config=self.eval_cfg,
                    train_config=self.train_cfg,
                    global_step=self.global_step,
                    epoch=epoch,
                    eval_name="fixed_eval",
                )
                if self.eval_cfg.enable_free_run_eval:
                    free_run_eval_result = render_eval_samples(
                        accelerator=self.accelerator,
                        model=self.model,
                        processor=self.processor,
                        speaker_name=self.train_cfg.speaker_name,
                        eval_rows=self.fixed_eval_rows,
                        output_dir=self.paths.free_run_eval_audio_dir / ckpt_dir.name,
                        eval_config=self.eval_cfg,
                        train_config=self.train_cfg,
                        global_step=self.global_step,
                        epoch=epoch,
                        eval_name="free_run_eval",
                    )

            for eval_name, eval_result in [("fixed_eval", fixed_eval_result), ("free_run_eval", free_run_eval_result)]:
                if eval_result is None:
                    continue
                eval_summary = eval_result["summary"]
                prefix = f"{eval_name}_"
                summary.update({f"{prefix}{key}": value for key, value in eval_summary.items()})
                self.accelerator.log(
                    {
                        f"{eval_name}/mean_duration_ratio": eval_summary["mean_duration_ratio"],
                        f"{eval_name}/max_duration_ratio": eval_summary["max_duration_ratio"],
                        f"{eval_name}/mean_abs_duration_error": eval_summary["mean_abs_duration_error"],
                        f"{eval_name}/cap_hit_rate": eval_summary["cap_hit_rate"],
                        f"{eval_name}/num_failed_samples": eval_summary["num_failed_samples"],
                        f"{eval_name}/mean_peak": eval_summary["mean_peak"],
                        f"{eval_name}/max_peak": eval_summary["max_peak"],
                        f"{eval_name}/mean_clipped_frac": eval_summary["mean_clipped_frac"],
                        f"{eval_name}/max_clipped_frac": eval_summary["max_clipped_frac"],
                        f"{eval_name}/mean_hf_noise_ratio": eval_summary["mean_hf_noise_ratio"],
                        f"{eval_name}/mean_voiced_f0_delta_p95": eval_summary["mean_voiced_f0_delta_p95"],
                        f"{eval_name}/qc_score": eval_summary["qc_score"],
                    },
                    step=self.global_step,
                )

            selection_eval_name = "free_run_eval" if free_run_eval_result is not None else "fixed_eval"
            selection_result = free_run_eval_result if free_run_eval_result is not None else fixed_eval_result
            if selection_result is not None:
                selection_summary = selection_result["summary"]
                selection_score = float(selection_summary["qc_score"])
                if self.tracker.best_qc_score is None or selection_score < self.tracker.best_qc_score:
                    self.tracker.best_qc_score = selection_score
                    self.tracker.best_checkpoint_path = str(ckpt_dir)
                    self.tracker.best_epoch = int(epoch)
                    self.tracker.best_eval_name = selection_eval_name
                    self.tracker.epochs_since_improvement = 0
                    write_best_checkpoint_record(
                        self.paths.output_dir,
                        best_epoch=self.tracker.best_epoch,
                        best_checkpoint_path=self.tracker.best_checkpoint_path,
                        best_qc_score=self.tracker.best_qc_score,
                        best_eval_name=self.tracker.best_eval_name,
                    )
                else:
                    self.tracker.epochs_since_improvement += 1

                summary["best_qc_score"] = self.tracker.best_qc_score
                summary["best_checkpoint_path"] = self.tracker.best_checkpoint_path
                summary["best_epoch"] = self.tracker.best_epoch
                summary["best_eval_name"] = self.tracker.best_eval_name
                summary["epochs_since_improvement"] = self.tracker.epochs_since_improvement
                if selection_eval_name == "free_run_eval":
                    if float(selection_summary["max_duration_ratio"]) > 1.8:
                        should_stop = True
                        early_stop_reason = "free_run_max_duration_ratio_exceeded"
                    elif float(selection_summary["cap_hit_rate"]) > 0.0:
                        should_stop = True
                        early_stop_reason = "free_run_cap_hit_rate_nonzero"
                if self.tracker.epochs_since_improvement >= int(self.checkpoint_cfg.early_stop_patience):
                    should_stop = True
                    early_stop_reason = early_stop_reason or "qc_no_improvement"

            write_json(ckpt_dir / "train_summary.json", summary)
            append_metrics_csv(
                self.paths.epoch_csv,
                {
                    "epoch": epoch,
                    "epoch_loss": safe_round(summary.get("avg_epoch_loss"), 8),
                    "epoch_main_loss": safe_round(summary.get("avg_epoch_main_loss"), 8),
                    "epoch_sub_loss": safe_round(summary.get("avg_epoch_sub_loss"), 8),
                    "epoch_main_kd_loss": safe_round(summary.get("avg_epoch_main_kd_loss"), 8),
                    "epoch_sub_kd_loss": safe_round(summary.get("avg_epoch_sub_kd_loss"), 8),
                    "epoch_tokens": summary.get("epoch_tokens"),
                    "epoch_seconds": safe_round(summary.get("epoch_seconds"), 4),
                    "epoch_tokens_per_sec": safe_round(summary.get("epoch_tokens_per_sec"), 4),
                    "global_step_end": self.global_step,
                    "stage_name": current_stage,
                    "fixed_eval_max_duration_ratio": safe_round(summary.get("fixed_eval_max_duration_ratio"), 8),
                    "fixed_eval_qc_score": safe_round(summary.get("fixed_eval_qc_score"), 8),
                    "free_run_eval_max_duration_ratio": safe_round(summary.get("free_run_eval_max_duration_ratio"), 8),
                    "free_run_eval_qc_score": safe_round(summary.get("free_run_eval_qc_score"), 8),
                },
            )
            if self.train_cfg.save_plots_every_epoch:
                self.plotter.plot_step_metrics(self.paths.step_csv)
                self.plotter.plot_epoch_metrics(self.paths.epoch_csv)

        self.tracker.mark_progress(
            next_epoch=epoch + 1,
            next_batch_in_epoch=0,
            global_step=self.global_step,
            last_step_checkpoint=self.latest_step_checkpoint,
            last_reason="epoch_completed",
            wandb_run_id=self.wandb_run_id,
            wandb_run_name=self.wandb_run_name,
            best_qc_score=self.tracker.best_qc_score,
            best_checkpoint_path=self.tracker.best_checkpoint_path,
            best_epoch=self.tracker.best_epoch,
            best_eval_name=self.tracker.best_eval_name,
            epochs_since_improvement=self.tracker.epochs_since_improvement,
        )
        self.accelerator.wait_for_everyone()
        stop_flag = torch.tensor([1 if should_stop else 0], device=self.accelerator.device)
        should_stop = bool(int(self.accelerator.gather(stop_flag).max().item()))
        if should_stop:
            self.accelerator.print(f"Early stop triggered at epoch={epoch} | reason={early_stop_reason}")
        return should_stop

    def _finalize_run(self):
        if self.accelerator is not None and self.accelerator.is_main_process:
            self.plotter.plot_step_metrics(self.paths.step_csv)
            self.plotter.plot_epoch_metrics(self.paths.epoch_csv)
        if self.speaker_row_hook is not None:
            self.speaker_row_hook.remove()
        if self.accelerator is not None:
            self.accelerator.end_training()
