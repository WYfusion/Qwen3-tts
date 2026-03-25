# coding=utf-8

CUSTOM_SPEAKER_ID = 3000
SPEAKER_SLOT_INDEX = 6

REQUIRED_FIXED_EVAL_FIELDS = {
    "sample_id",
    "text",
    "language",
    "speaker",
    "audio",
    "ref_audio",
    "target_seconds",
    "target_code_frames",
}

STEP_PLOT_KEYS = [
    ("loss", "train_loss_vs_step.png"),
    ("main_loss", "train_main_loss_vs_step.png"),
    ("sub_loss", "train_sub_loss_vs_step.png"),
    ("main_kd_loss", "train_main_kd_loss_vs_step.png"),
    ("sub_kd_loss", "train_sub_kd_loss_vs_step.png"),
    ("lr", "learning_rate_vs_step.png"),
    ("grad_norm", "grad_norm_vs_step.png"),
    ("tokens_per_sec", "tokens_per_sec_vs_step.png"),
    ("codec_tokens", "codec_tokens_vs_step.png"),
]

EPOCH_PLOT_KEYS = [
    ("epoch_loss", "epoch_loss.png"),
    ("epoch_main_loss", "epoch_main_loss.png"),
    ("epoch_sub_loss", "epoch_sub_loss.png"),
    ("epoch_main_kd_loss", "epoch_main_kd_loss.png"),
    ("epoch_sub_kd_loss", "epoch_sub_kd_loss.png"),
    ("epoch_tokens_per_sec", "epoch_tokens_per_sec.png"),
    ("fixed_eval_max_duration_ratio", "fixed_eval_max_duration_ratio.png"),
    ("fixed_eval_qc_score", "fixed_eval_qc_score.png"),
    ("free_run_eval_max_duration_ratio", "free_run_eval_max_duration_ratio.png"),
    ("free_run_eval_qc_score", "free_run_eval_qc_score.png"),
]

