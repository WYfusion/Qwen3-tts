# Qwen3-TTS 单说话人 SFT 工程化训练指南

## 1. 训练入口

当前官方训练入口为：

```bash
python -m finetuning.scr_sft.cli ...
```

兼容入口仍然保留，但它们现在只做参数转发：

- `python finetuning/sft_12hz_with_metrics.py ...`
- `python finetuning/sft_12hz.py ...`
- `python finetuning/diagnose_fixed_eval_audio.py ...`
- `python finetuning/plot_qwen3tts_metrics.py ...`

这意味着训练逻辑、评估逻辑、导出逻辑和诊断逻辑都只维护一份正式实现，避免旧单体脚本继续膨胀。

## 2. 包结构

```text
finetuning/
  __init__.py
  scr_sft/
    __init__.py
    cli.py
    args.py
    config.py
    constants.py
    io_utils.py
    data.py
    model_ops.py
    recipes.py
    losses.py
    tracking.py
    state.py
    audio_qc.py
    eval_audio.py
    export.py
    plots.py
    diagnostics.py
    trainer.py
```

## 3. 模块职责地图

- `cli.py`
  - 官方 CLI 入口
  - 解析参数并启动 `SFTTrainer`
- `args.py`
  - 统一维护全部 `argparse` 参数
  - 保证旧 CLI flag 兼容
- `config.py`
  - 将参数收口为 dataclass 配置对象
  - 包含 `TrainConfig`、`EvalConfig`、`LoggingConfig`、`CheckpointConfig`、`RunPaths`
- `constants.py`
  - 固定常量，如 `CUSTOM_SPEAKER_ID=3000`
  - 绘图指标键定义
- `io_utils.py`
  - JSON / JSONL / CSV 读写
  - 路径创建与安全序列化
  - 指标 CSV 追加写入
- `data.py`
  - `TTSDataset`
  - `fixed_eval_set.jsonl` 生成
  - 数据集统计与 epoch dataloader 构建
- `model_ops.py`
  - student / teacher 模型加载
  - flash-attn 自动回退
  - custom speaker row 初始化
  - talker forward 相关 helper
- `recipes.py`
  - staged recipe 定义
  - 参数分组与冻结/解冻切换
  - speaker row 梯度掩码
- `losses.py`
  - main CE / sub CE / KD 组合逻辑
- `tracking.py`
  - `wandb` / tracker 初始化
  - run name、group、resume 相关逻辑
- `state.py`
  - 训练状态保存与恢复
  - step checkpoint / epoch checkpoint
  - 异常 batch 落盘
- `audio_qc.py`
  - 训练前音频质检
  - 生成 `audio_qc_report.json/csv`
- `eval_audio.py`
  - capped eval / free-run eval
  - manifest 生成
  - QC 分数计算
  - wandb 音频表记录
- `export.py`
  - custom voice inference checkpoint 导出
  - `best_checkpoint.json` 维护
- `plots.py`
  - 训练指标曲线绘制
  - 独立 plot CLI 后端
- `diagnostics.py`
  - 离线诊断已有试听目录
  - 输出 `diagnosis_report.json/csv`
- `trainer.py`
  - 训练总编排器 `SFTTrainer`
  - 串联 setup / train / eval / export / finalize

## 4. 模型与训练策略

当前单说话人 SFT 方案保持以下核心行为不变：

- 基座模型从 `Qwen3-TTS-12Hz-1.7B-Base` 初始化
- 自定义 speaker slot 固定为 `3000`
- 训练配方默认是 `staged_stable_sft`
- 保留 `legacy_full_sft` 作为兼容配方
- 训练中同时进行 capped eval 和 free-run eval
- 选模只看 free-run QC，不看训练 loss

当前默认策略的关键点：

- Stage 1 只开放高层 `talker` 模块与自定义 speaker row
- Stage 2 再解冻 `talker.model.layers.20-27.*` 与 `talker.code_predictor.*`
- teacher 固定为 base 模型，用于保持时长、EOS 和高层生成先验
- 导出 checkpoint 时不修改 `generation_config.json` 默认采样逻辑

## 5. 标准训练流程

### 5.1 生成原始 JSONL

```bash
cd finetuning

python script/raw2jsonl.py \
  --audio_dir ../assets/BZNSYP_24k/Wave \
  --prosody_dir ../assets/BZNSYP/ProsodyLabeling \
  --ref_audio ../assets/BZNSYP_24k/ref.wav \
  --out_dir ../assets/BZNSYP_24k/ft_data \
  --val_size 100 \
  --test_size 100 \
  --seed 42 \
  --relative_to ../
```

### 5.2 提取 `audio_codes`

```bash
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path ../Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl ../assets/BZNSYP_24k/ft_data/train_raw.jsonl \
  --output_jsonl ../assets/BZNSYP_24k/codec/train_with_codes.jsonl
```

### 5.3 启动训练

```bash
python -m finetuning.scr_sft.cli \
  --init_model_path ./Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path finetuning/exp/output_bznsyp_1p7b_sft \
  --train_jsonl ./assets/BZNSYP_24k/codec/train_with_codes.jsonl \
  --speaker_name bznsyp_female \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --training_recipe staged_stable_sft \
  --stage1_epochs 1 \
  --stage2_epochs 3 \
  --main_kd_weight 0.1 \
  --sub_kd_weight 0.05 \
  --kd_temperature 2.0 \
  --fixed_eval_num_samples 4 \
  --enable_free_run_eval \
  --free_run_eval_max_new_tokens 8192 \
  --peak_warn_threshold 0.99 \
  --clipped_frac_warn_threshold 1e-6 \
  --hf_noise_warn_threshold 0.12 \
  --voiced_f0_delta_warn_threshold 180 \
  --early_stop_patience 1 \
  --save_training_state_steps 100
```

### 5.4 只做 dry-run

```bash
python -m finetuning.scr_sft.cli \
  --init_model_path ./Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path finetuning/exp/output_bznsyp_1p7b_sft \
  --train_jsonl ./assets/BZNSYP_24k/codec/train_with_codes.jsonl \
  --speaker_name bznsyp_female \
  --dry_run
```

### 5.5 只生成音频质检报告

```bash
python -m finetuning.scr_sft.cli \
  --init_model_path ./Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path finetuning/exp/output_bznsyp_1p7b_sft \
  --train_jsonl ./assets/BZNSYP_24k/codec/train_with_codes.jsonl \
  --speaker_name bznsyp_female \
  --audio_qc_report_only
```

## 6. 训练期评估

训练期包含两条评估路径：

- `fixed_eval`
  - 受控试听
  - 使用目标时长派生的 `max_new_tokens`
- `free_run_eval`
  - 默认生成路径回归验证
  - 不按目标时长限长，只保留全局安全上限

训练期统一记录这些指标：

- `decoded_seconds`
- `duration_ratio`
- `peak`
- `clipped_frac`
- `hf_noise_ratio`
- `voiced_f0_delta_p95`

当前放行阈值：

- `max_duration_ratio < 1.5`
- `peak < 0.99`
- `clipped_frac <= 1e-6`

当前早停规则：

- `free_run_max_duration_ratio > 1.8`
- `free_run_cap_hit_rate > 0`
- QC 连续 `early_stop_patience` 个 epoch 无提升

## 7. 产物目录

```text
output_model_path/
  checkpoint-epoch-0/
  checkpoint-epoch-1/
  best_checkpoint.json
  training_state/
  logs/
    audio_qc/
    metrics/
    plots/
    fixed_eval_audio/
    free_run_eval_audio/
    anomaly_batches/
```

重点文件说明：

- `logs/audio_qc/audio_qc_report.json`
  - 训练前全量音频质检汇总
- `logs/metrics/train_step_metrics.csv`
  - step 级训练指标
- `logs/metrics/train_epoch_metrics.csv`
  - epoch 级训练指标
- `logs/fixed_eval_audio/<checkpoint>/manifest.json`
  - 受控试听 manifest
- `logs/free_run_eval_audio/<checkpoint>/manifest.json`
  - 默认生成路径 manifest
- `best_checkpoint.json`
  - 当前 free-run QC 最优 checkpoint

## 8. 诊断与绘图

### 8.1 诊断已有试听目录

```bash
python -m finetuning.scr_sft.diagnostics \
  --fixed_eval_dir finetuning/exp/output_bznsyp_1p7b_sft/logs/free_run_eval_audio/checkpoint-epoch-0 \
  --fixed_eval_jsonl finetuning/exp/output_bznsyp_1p7b_sft/logs/fixed_eval_set.jsonl
```

### 8.2 绘制训练曲线

```bash
python -m finetuning.scr_sft.plots \
  --metrics_dir finetuning/exp/output_bznsyp_1p7b_sft/logs/metrics \
  --plots_dir finetuning/exp/output_bznsyp_1p7b_sft/logs/plots
```

## 9. 推理验证

导出后建议直接走默认 `generate_custom_voice()` 路径验证，不显式传时长控制参数。

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    "finetuning/exp/output_bznsyp_1p7b_sft/checkpoint-epoch-0",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = tts.generate_custom_voice(
    text="今天天气不错，我们下午一起去公园散步吧。",
    speaker="bznsyp_female",
)
sf.write("finetuning_verify.wav", wavs[0], sr)
```

## 10. 设计原则

本工程化重构的目标不是改变训练语义，而是保证：

- 单一正式训练入口
- 单一正式 trainer 编排实现
- 评估、导出、绘图、诊断各自独立
- CLI、目录结构、manifest 字段保持兼容
- 后续新增能力时可以按模块扩展，而不是继续向单个脚本堆功能
