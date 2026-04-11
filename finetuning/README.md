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
    benchmark_eval.py
    export.py
    plots.py
    diagnostics.py
    seed_tts_eval.py
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
- `benchmark_eval.py`
  - 训练期 `seed-tts-eval` 桥接
  - Base baseline 缓存
  - benchmark subset 分层抽样
  - WER / ASV 解析与 benchmark 选模
- `export.py`
  - custom voice inference checkpoint 导出
  - `best_checkpoint.json` 维护
- `plots.py`
  - 训练指标曲线绘制
  - 独立 plot CLI 后端
- `diagnostics.py`
  - 离线诊断已有试听目录
  - 输出 `diagnosis_report.json/csv`
- `seed_tts_eval.py`
  - 为 `seed-tts-eval` 准备离线评测音频
  - 生成 `generated/*.wav`、`meta.lst` 与评测 manifest
- `trainer.py`
  - 训练总编排器 `SFTTrainer`
  - 串联 setup / train / eval / export / finalize

## 4. 模型与训练策略

当前单说话人 SFT 方案保持以下核心行为：

- 基座模型从 `Qwen3-TTS-12Hz-1.7B-Base` 初始化
- 自定义 speaker slot 固定为 `3000`
- 训练配方默认是 `staged_benchmark_aligned_sft`
- 保留 `legacy_full_sft` 作为兼容配方
- 训练中同时进行 capped eval 和 free-run eval
- 保留 QC 安全门，但选模不再只看 4 条试听代理指标

当前默认策略的关键点：

- Stage 1 训练 `speaker_row + head + upper_layers`
- Stage 2 在此基础上加入 `mid_layers + code_predictor`
- Stage 3 再加入 `lower_mid_layers`
- teacher 固定为 base 模型，但 KD 权重按 stage 缩放：`2.0x -> 1.0x -> 0.6x`
- custom speaker row 不再只用第一条样本初始化，而是用多参考均值初始化
- 每个 epoch 额外跑一份 benchmark subset，并同时记录 `WER + ASV`
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
python finetuning/prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path ./Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl ./assets/BZNSYP_24k/ft_data/train_raw.jsonl \
  --output_jsonl ./assets/BZNSYP_24k/codec/train_with_codes.jsonl
```

```bash
python finetuning/prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path ./Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl ./assets/BZNSYP_24k/ft_data/test_raw.jsonl \
  --output_jsonl ./assets/BZNSYP_24k/codec/test_with_codes.jsonl
```

```bash
python finetuning/prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path ./Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl ./assets/BZNSYP_24k/ft_data/val_raw.jsonl \
  --output_jsonl ./assets/BZNSYP_24k/codec/val_with_codes.jsonl
```

### 5.3 使用 uv 同步训练依赖
在使用 `uv` 时，优先执行：

```bash
uv sync
```

如果还需要系统级音频工具，例如 `sox` 和 `ffmpeg`，再执行：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg sox libsox-fmt-all
```

### 5.4 启动训练

```bash
python -m finetuning.scr_sft.cli \
  --init_model_path ./Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path finetuning/exp/output_bznsyp_1p7b_sft-4-11 \
  --train_jsonl ./assets/BZNSYP_24k/codec/train_with_codes.jsonl \
  --speaker_name bznsyp_female \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --training_recipe staged_benchmark_aligned_sft \
  --stage1_epochs 1 \
  --stage2_epochs 2 \
  --stage3_epochs 2 \
  --main_kd_weight 0.1 \
  --sub_kd_weight 0.05 \
  --kd_temperature 2.0 \
  --speaker_init_num_samples 16 \
  --fixed_eval_num_samples 24 \
  --enable_free_run_eval \
  --free_run_eval_max_new_tokens 8192 \
  --benchmark_eval_jsonl ./assets/BZNSYP_24k/codec/test_with_codes.jsonl \
  --benchmark_eval_num_samples 48 \
  --benchmark_eval_every_epochs 1 \
  --seed_tts_eval_root ./seed-tts-eval \
  --seed_tts_eval_python python3 \
  --benchmark_eval_device cuda:0 \
  --sim_finetune_checkpoint ./seed-tts-eval/weight/wavlm_large_finetune.pth \
  --peak_warn_threshold 0.99 \
  --clipped_frac_warn_threshold 1e-6 \
  --hf_noise_warn_threshold 0.12 \
  --voiced_f0_delta_warn_threshold 180 \
  --early_stop_patience 2 \
  --save_training_state_steps 100
```

### 5.5 只做 dry-run

```bash
python -m finetuning.scr_sft.cli \
  --init_model_path ./Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path finetuning/exp/output_bznsyp_1p7b_sft \
  --train_jsonl ./assets/BZNSYP_24k/codec/train_with_codes.jsonl \
  --speaker_name bznsyp_female \
  --dry_run
```

### 5.6 只生成音频质检报告

```bash
python -m finetuning.scr_sft.cli \
  --init_model_path ./Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path finetuning/exp/output_bznsyp_1p7b_sft \
  --train_jsonl ./assets/BZNSYP_24k/codec/train_with_codes.jsonl \
  --speaker_name bznsyp_female \
  --audio_qc_report_only
```

## 6. 训练期评估

训练期包含三条评估路径：

- `fixed_eval`
  - 受控试听
  - 使用目标时长派生的 `max_new_tokens`
- `free_run_eval`
  - 默认生成路径回归验证
  - 不按目标时长限长，只保留全局安全上限
- `benchmark_eval`
  - 通过 `seed-tts-eval` 跑 WER / ASV
  - 先缓存 Base 全量基线，再抽取难度分层的 subset
  - 每个 epoch 对导出 checkpoint 跑 subset，用于对齐外部目标

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

当前选模规则：

- `best_safe_checkpoint`
  - 仍由 free-run / fixed-eval 的 `qc_score` 驱动
  - 负责安全回退
- `best_benchmark_checkpoint`
  - 必须先满足 free-run 安全门
  - 只在 `WER < Base` 且 `ASV > Base` 时才有资格成为 benchmark best
  - tie-break 依次看 `benchmark_wer`、`benchmark_asv`、`free_run_qc_score`
- `best_checkpoint.json`
  - 同时写入 `best_safe_checkpoint`、`best_benchmark_checkpoint`
  - 若 benchmark 目标达成，则默认主部署点为 benchmark best；否则回退 safe best

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
    benchmark_eval/
    anomaly_batches/
```

重点文件说明：

- `logs/audio_qc/audio_qc_report.json`
  - 训练前全量音频质检汇总
- `logs/metrics/train_step_metrics.csv`
  - step 级训练指标
- `logs/metrics/train_epoch_metrics.csv`
  - epoch 级训练指标
- `logs/metrics/benchmark_epoch_metrics.csv`
  - epoch 级 benchmark WER / ASV / objective
- `logs/fixed_eval_audio/<checkpoint>/manifest.json`
  - 受控试听 manifest
- `logs/free_run_eval_audio/<checkpoint>/manifest.json`
  - 默认生成路径 manifest
- `logs/benchmark_eval/benchmark_subset_manifest.json`
  - 本轮训练使用的 benchmark subset 清单
- `best_checkpoint.json`
  - 同时记录 safe best、benchmark best 与主部署 checkpoint

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

### 9.1 使用 `seed-tts-eval` 做离线客观评测

对于 `finetuning/exp/output_bznsyp_1p7b_sft_3-25/checkpoint-epoch-1`，推荐先生成一份专供 `seed-tts-eval` 使用的评测目录，再分别计算 WER 和 SIM。

#### 9.1.1 生成评测音频与 meta

```bash
python finetuning/prepare_seed_tts_eval.py \
  --checkpoint_dir finetuning/exp/output_bznsyp_1p7b_sft_3-25/checkpoint-epoch-1 \
  --eval_jsonl assets/BZNSYP_24k/codec/test_with_codes.jsonl \
  --speaker_name bznsyp_female \
  --output_dir finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-1
```
```bash
python finetuning/prepare_seed_tts_eval.py \
  --checkpoint_dir finetuning/exp/output_bznsyp_1p7b_sft_3-25/checkpoint-epoch-0 \
  --eval_jsonl assets/BZNSYP_24k/codec/test_with_codes.jsonl \
  --speaker_name bznsyp_female \
  --output_dir finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-0
```
如果你要评估未 SFT 的 `Qwen3-TTS-12Hz-1.7B-Base`，现在也可以直接复用同一个入口：

```bash
python finetuning/prepare_seed_tts_eval.py \
  --checkpoint_dir Qwen3-TTS-12Hz-1.7B-Base \
  --eval_jsonl assets/BZNSYP_24k/codec/test_with_codes.jsonl \
  --output_dir finetuning/exp/base_seed_tts_eval
```

这里会自动识别 base 模型并切到 `generate_voice_clone()` 路径，默认使用每条样本里的 `ref_audio` 做 voice clone prompt，且为了最小改动采用 `x_vector_only_mode=True`。这意味着：

- 不需要额外提供 `--speaker_name`
- 不需要修改现有 `test_with_codes.jsonl` 数据格式
- 后续 WER / SIM 命令与 SFT 模型完全一致，只需把目录换成 `finetuning/exp/base_seed_tts_eval`

这里推荐直接传入已经编码好的 `assets/BZNSYP_24k/codec/test_with_codes.jsonl`。这样评测脚本会优先使用 `audio_codes` 的帧数作为目标长度参考，避免再对评测集音频做一次运行时编码。

该命令会固定使用：

- `language="Chinese"`
- `device="cuda:0"`
- `dtype=torch.bfloat16`
- `attn_implementation="flash_attention_2"`
- deterministic 解码：`do_sample=False`、`top_k=1`、`top_p=1.0`、`temperature=1.0`
- 动态长度上限：优先走 `max_new_tokens = min(256, round(target_code_frames * 2.0))`
- 如果输入 jsonl 不含 `audio_codes`，则回退为 `max_new_tokens = min(256, round(target_seconds * 12.5 * 2.0))`

输出目录结构如下：

```text
seed_tts_eval/checkpoint-epoch-1/
  generated/
    000761.wav
    ...
  meta.lst
  manifest.json
  wav_res_ref_text
  wer.raw.txt
  wer.summary.txt
  sim.raw.txt
  sim.summary.txt
```

其中：

- `meta.lst` 采用 `utt|infer_text|prompt_wav` 三列格式
- `manifest.json` 额外记录 `target_audio`、`ref_audio`、`target_code_frames`、`target_seconds`、生成参数和输出 wav 路径

如需和 `checkpoint-epoch-0` 做对照，直接把 `--checkpoint_dir` 和 `--output_dir` 替换成 `checkpoint-epoch-0` 即可。

如果你想把 base 模型和 SFT 模型放到同一目录层级下做横向对比，也可以约定输出成：

```text
finetuning/exp/
  base_seed_tts_eval/
  output_bznsyp_1p7b_sft_3-25/
    seed_tts_eval/
      checkpoint-epoch-0/
      checkpoint-epoch-1/
```

#### 9.1.2 安装 `seed-tts-eval` 依赖

WER 侧依赖：

```bash
pip install -r seed-tts-eval/requirements.txt
```

SIM 侧依赖请参考：

- `seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification/README.md`
- 并提前准备 `seed-tts-eval/weight/wavlm_large_finetune.pth`
- SIM 评估实际依赖两类权重：
  1. `seed-tts-eval/weight/wavlm_large_finetune.pth`
  2. upstream `wavlm_large.pt`，默认缓存到 `seed-tts-eval/weight/huggingface`

建议将“生成 wav”和“跑 WER/SIM”放在两个独立环境中，避免 `speaker_verification` 旧依赖污染当前 Qwen3-TTS 训练环境。

#### 9.1.3 计算 WER

```bash
python seed-tts-eval/get_wav_res_ref_text.py \
  finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-1/meta.lst \
  finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-1/generated \
  finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-1/wav_res_ref_text

python seed-tts-eval/prepare_ckpt.py

python seed-tts-eval/run_wer.py \
  finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-1/wav_res_ref_text \
  finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-1/wer.raw.txt \
  zh

python seed-tts-eval/average_wer.py \
  finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-1/wer.raw.txt \
  finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-1/wer.summary.txt
```
```bash
python seed-tts-eval/get_wav_res_ref_text.py \
  finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-0/meta.lst \
  finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-0/generated \
  finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-0/wav_res_ref_text

python seed-tts-eval/prepare_ckpt.py

python seed-tts-eval/run_wer.py \
  finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-0/wav_res_ref_text \
  finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-0/wer.raw.txt \
  zh

python seed-tts-eval/average_wer.py \
  finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-0/wer.raw.txt \
  finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-0/wer.summary.txt
```
对 base 模型，目录直接替换成 `finetuning/exp/base_seed_tts_eval`：

```bash
python seed-tts-eval/get_wav_res_ref_text.py \
  finetuning/exp/base_seed_tts_eval/meta.lst \
  finetuning/exp/base_seed_tts_eval/generated \
  finetuning/exp/base_seed_tts_eval/wav_res_ref_text

python seed-tts-eval/prepare_ckpt.py

python seed-tts-eval/run_wer.py \
  finetuning/exp/base_seed_tts_eval/wav_res_ref_text \
  finetuning/exp/base_seed_tts_eval/wer.raw.txt \
  zh

python seed-tts-eval/average_wer.py \
  finetuning/exp/base_seed_tts_eval/wer.raw.txt \
  finetuning/exp/base_seed_tts_eval/wer.summary.txt
```
其中中文 WER 使用 `funasr` 的 `paraformer-zh` 别名，对应的实际 ModelScope 模型是 `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`。脚本现在会优先复用 `seed-tts-eval/weight/modelscope` 下已有缓存；如果你当前是 Python 3.12 环境且 `hydra` 导入时报 dataclass mutable default 错误，请改用单独的 Python 3.10/3.11 评估环境。

`run_wer.py` 默认会复用 `seed-tts-eval/weight/huggingface` 与 `seed-tts-eval/weight/modelscope` 下的已下载权重，并默认使用 `https://hf-mirror.com`。如果你需要显式指定，也可以追加：

```bash
  --hf_cache_dir seed-tts-eval/weight/huggingface \
  --modelscope_cache_dir seed-tts-eval/weight/modelscope \
  --hf_endpoint https://hf-mirror.com
```

#### 9.1.4 计算 SIM

```bash
python seed-tts-eval/prepare_ckpt.py \
  --prepare_sim \
  --sim_model_name wavlm_large \
  --sim_finetune_checkpoint ./seed-tts-eval/weight/wavlm_large_finetune.pth

python seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification/verification_pair_list_v2.py \
  finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-1/wav_res_ref_text \
  --model_name wavlm_large \
  --checkpoint ./seed-tts-eval/weight/wavlm_large_finetune.pth \
  --scores finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-1/sim.raw.txt \
  --wav1_start_sr 0 \
  --wav2_start_sr 0 \
  --wav1_end_sr -1 \
  --wav2_end_sr -1 \
  --device cuda:0

python seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification/average.py \
  finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-1/sim.raw.txt \
  finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-1/sim.summary.txt
```
```bash
python seed-tts-eval/prepare_ckpt.py \
  --prepare_sim \
  --sim_model_name wavlm_large \
  --sim_finetune_checkpoint ./seed-tts-eval/weight/wavlm_large_finetune.pth

python seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification/verification_pair_list_v2.py \
  finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-0/wav_res_ref_text \
  --model_name wavlm_large \
  --checkpoint ./seed-tts-eval/weight/wavlm_large_finetune.pth \
  --scores finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-0/sim.raw.txt \
  --wav1_start_sr 0 \
  --wav2_start_sr 0 \
  --wav1_end_sr -1 \
  --wav2_end_sr -1 \
  --device cuda:0

python seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification/average.py \
  finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-0/sim.raw.txt \
  finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-0/sim.summary.txt
```
对 base 模型同样只替换目录：

```bash
python seed-tts-eval/prepare_ckpt.py \
  --prepare_sim \
  --sim_model_name wavlm_large \
  --sim_finetune_checkpoint ./seed-tts-eval/weight/wavlm_large_finetune.pth

python seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification/verification_pair_list_v2.py \
  finetuning/exp/base_seed_tts_eval/wav_res_ref_text \
  --model_name wavlm_large \
  --checkpoint ./seed-tts-eval/weight/wavlm_large_finetune.pth \
  --scores finetuning/exp/base_seed_tts_eval/sim.raw.txt \
  --wav1_start_sr 0 \
  --wav2_start_sr 0 \
  --wav1_end_sr -1 \
  --wav2_end_sr -1 \
  --device cuda:0

python seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification/average.py \
  finetuning/exp/base_seed_tts_eval/sim.raw.txt \
  finetuning/exp/base_seed_tts_eval/sim.summary.txt
```

`verification_pair_list_v2.py` 现在默认会优先复用 `seed-tts-eval/weight/huggingface` 下的 repo-local upstream cache，不再依赖每次运行都在线拉取 `s3prl/s3prl`。如需显式指定本地 upstream ckpt，可追加：

```bash
  --upstream_ckpt /abs/path/to/wavlm_large.pt
```

如果你在一台可联网 Ubuntu 上已经跑通，想把缓存拷到另一台离线或网络受限机器，至少复制以下目录/文件：

```bash
seed-tts-eval/weight/huggingface/
seed-tts-eval/weight/wavlm_large_finetune.pth
```

只要 upstream `wavlm_large.pt` 已在上述 Hugging Face cache 中，SIM 初始化就会优先命中本地缓存；如果缓存缺失且网络不可用，脚本会在模型初始化阶段直接报错退出，而不是在每个样本上重复卡住。

#### 9.1.5 结果解释建议

- `wer.summary.txt` 给出整体 WER
- `sim.summary.txt` 给出平均 speaker similarity
- 当前训练日志里 `checkpoint-epoch-0` 的内部 fixed-eval QC 优于 `checkpoint-epoch-1`，建议两者都跑同一套 `seed-tts-eval` 流程做外部对照
- README 中推荐的流程不依赖 upstream `cal_wer.sh` / `cal_sim.sh`，因为它们偏 Linux / 多卡，且本地 `cal_wer.sh` 末尾汇总路径并不适合直接照搬

#### 9.1.6 三模型 WER / SIM 实测对比与 `scr_sft` 流程分析

下面给出基于本地实际结果的三模型正式对比。这里的“三模型”分别指：

- Base：`Qwen3-TTS-12Hz-1.7B-Base`
- SFT epoch-0：`finetuning/exp/output_bznsyp_1p7b_sft_3-25/checkpoint-epoch-0`
- SFT epoch-1：`finetuning/exp/output_bznsyp_1p7b_sft_3-25/checkpoint-epoch-1`

结果来源固定为以下目录，而不是终端粘贴的命令文本：

- Base：`finetuning/exp/base_seed_tts_eval`
- SFT：`finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-0`
- SFT：`finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-1`

结论先行：

- 内容一致性排序：`epoch-1 > base > epoch-0`
- 说话人相似度排序：`epoch-1 > epoch-0 > base`
- 这次 SFT 的收益不大，但方向成立，呈现出比较典型的“两阶段趋势”：
  - `stage1` 先把音色往目标说话人方向拉近
  - `stage2` 再把内容一致性拉回，并最终略微超过 base

**评测协议**

- WER 含义：基于 ASR 转写结果计算字词错误率，`越低越好`，主要衡量文本内容是否被正确说出。
- SIM / ASV 含义：基于说话人验证模型计算 speaker similarity，`越高越好`，主要衡量生成音频与参考音色的接近程度。
- `ASV-var` 含义：speaker similarity 在测试集上的波动程度，`越低越稳定`。
- WER 工具链：`seed-tts-eval/get_wav_res_ref_text.py` -> `seed-tts-eval/run_wer.py zh` -> `seed-tts-eval/average_wer.py`。
- 中文 ASR 实际使用的是 FunASR 的 `paraformer-zh` 别名，对应 ModelScope 模型 `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`。
- SIM 工具链：`seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification/verification_pair_list_v2.py` -> `average.py`。
- SIM 依赖的 speaker verification 权重是 `wavlm_large` upstream 与 `seed-tts-eval/weight/wavlm_large_finetune.pth`。
- 外部 benchmark 使用 `assets/BZNSYP_24k/codec/test_with_codes.jsonl`，样本数固定为 `100`。
- 这里的可比性是“同数据、同下游指标、同离线评测链路”的任务级可比，而不是完全同结构同推理范式的严格 apples-to-apples：
  - Base 评测走的是 `generate_voice_clone(..., ref_audio=..., x_vector_only_mode=True)`
  - SFT checkpoint 评测走的是 `generate_custom_voice(...)`

**实测结果**

| 模型 | 推理方式 | WER ↓ | 相对 Base 变化 | ASV ↑ | 相对 Base 变化 | ASV-var |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base | voice clone | 3.374% | 0.000 | 0.790 | 0.000 | 0.002 |
| SFT epoch-0 | custom voice | 3.477% | +0.103 | 0.792 | +0.002 | 0.002 |
| SFT epoch-1 | custom voice | 3.347% | -0.027 | 0.794 | +0.004 | 0.002 |

进一步看三组之间的变化：

- Base -> epoch-0：WER 上升 `0.103` 个百分点，说明内容一致性略退；ASV 提升 `0.002`，说明音色更接近目标说话人。
- Base -> epoch-1：WER 下降 `0.027` 个百分点，说明内容一致性略优于 base；ASV 提升 `0.004`，音色也略优于 base。
- epoch-0 -> epoch-1：WER 下降 `0.130` 个百分点，ASV 再提升 `0.002`，说明第二阶段确实在修复 stage1 的内容退化，同时继续改善音色。
- `ASV-var` 三组都在 `0.002` 左右，可以解读为整体稳定性差异不显著。

这组数值应被解读为“小幅正向收益”，而不是“显著跃升”。对于本身已经很强的 `Qwen3-TTS-12Hz-1.7B-Base`，单说话人 SFT 还能同时把 WER 和 ASV 都推到略优于 base，说明方向是对的；但提升量级不大，也说明当前训练策略更偏稳健微调，而不是激进突破。

如果看 100 条测试集上的逐样本变化，可以更清楚地看到这种“小幅、分布式增益”：

- `epoch-1` 相比 base，在 WER 上是 `2` 条改善、`95` 条不变、`3` 条变差，说明平均 WER 改善主要来自少数样本的纠错，而不是大面积重排。
- `epoch-0` 相比 base，在 WER 上是 `1` 条改善、`95` 条不变、`4` 条变差，说明 stage1 时内容确实更容易先退一点。
- `epoch-1` 相比 base，在 SIM 上是 `53` 条提升、`47` 条下降；`epoch-0` 是 `54` 条提升、`46` 条下降，说明 ASV 的提升更像是大量样本上的细小平均改善，而不是少量样本的大幅跃升。

如果进一步回答“是不是因为 benchmark 例子太难”，更合适的做法不是只盯整体均值，而是把 100 条句子按“主导难点”分桶。这里采用单标签人工分桶，优先级为：

- `人名地名 > 成语书面语 > 口语助词 > 普通句`
- 少数句子本来可以跨桶，但为了统计不重复，只保留一个主导标签

分桶后的整体结果如下：

| 分桶 | 样本数 | Base WER ↓ | epoch-1 WER ↓ | 相对 Base 变化 | Base ASV ↑ | epoch-1 ASV ↑ | Base WER 贡献占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 人名地名 | 34 | 4.228% | 4.249% | +0.021 | 0.7924 | 0.7984 | 42.6% |
| 成语书面语 | 22 | 5.405% | 5.819% | +0.413 | 0.7989 | 0.8033 | 35.2% |
| 口语助词 | 24 | 2.598% | 2.077% | -0.521 | 0.7764 | 0.7786 | 18.5% |
| 普通句 | 20 | 0.620% | 0.620% | +0.000 | 0.7920 | 0.7957 | 3.7% |

从这个角度看，可以得到几个更有解释力的结论：

- `人名地名 + 成语书面语` 只占 `56%` 的样本，却吃掉了 `77.8%` 的 Base WER。
- `普通句` 只贡献了 `3.7%` 的 Base WER，说明大部分常规句子本来就已经接近做穿。
- 这说明“样本难”确实是主要原因之一，但难点并不是“句子普遍太长”，而是“长尾词型集中”。
- 四个桶的平均文本长度分别约为：
  - `人名地名 = 16.88`
  - `成语书面语 = 15.27`
  - `口语助词 = 14.08`
  - `普通句 = 14.90`
- 长度差距并不大，所以当前 WER 的主要来源不是长句建模失败，而是专名、成语、文言感表达、低频书面词和口语同音词的长尾风险。

按桶展开看，现象会更清楚。

`人名地名` 桶：

- 这是典型的专名同音替换问题，内容并没有整体失控，但 ASR 很容易把名字、地名、专有词转成更高频的近音字。
- Base 的 WER 已经不低，`epoch-1` 基本打平，只从 `4.228%` 微调到 `4.249%`。
- 但 ASV 从 `0.7924` 提升到 `0.7984`，是四个桶里音色收益最明显的一档，说明 SFT 更像是在把音色往目标说话人方向拉，而不是系统性修复专名发音。
- 代表性误差包括：
  - `春磊 / 才副的 -> 春蕾 / 财富的`
  - `肖克 / 寸滩港 -> 逍客 / 寸摊岗`
  - `陈兴銮 -> 陈兴鸾`
  - `樊治容 -> 樊志荣`
- 这个桶里有 `10` 条样本在 Base、epoch-0、epoch-1 上都保持相同的非零 WER，说明一部分错误本来就非常顽固，不是继续多训一个 epoch 就能自然消掉的。

`成语书面语` 桶：

- 这是当前 benchmark 中最难的桶，Base WER 最高，达到 `5.405%`。
- `epoch-1` 在这个桶里没有修复内容一致性，反而上升到 `5.819%`；但 ASV 仍然从 `0.7989` 升到 `0.8033`。
- 这说明当前 SFT 在书面长尾词上，更倾向于保留或改善音色，而没有明显提升内容可辨识性。
- 代表性误差包括：
  - `泮池磅礴 -> 畔池磅礴`
  - `寅吃卯粮 -> 银吃卯粮`
  - `赌档 -> 赌当`
  - `陶鼎 -> 陶顶`
  - `右倾化 -> 右氢化`
- 这个桶里有 `11` 条样本在三模型上保持相同的非零 WER，而且 Base 到 `epoch-1` 实际只有 `1` 条发生变化，说明这类错误本质上就是 benchmark 的硬骨头，SFT 目前基本啃不动。

`口语助词` 桶：

- 这是当前 SFT 唯一出现明确内容收益的桶，WER 从 `2.598%` 降到 `2.077%`。
- 但这里也不能过度乐观，因为这部分改善并不是很多句子一起变好，而是主要集中在极少数样本上。
- 最典型的一条是：
  - `这能说他们是忍吗？`
  - Base 识别成 `这能说他们是人吗`
  - `epoch-1` 修成了 `这能说他们是忍吗`
- 也就是说，这个桶的改善是真实的，但更像“少数口语句被修正”，而不是“整类口语表达都被系统性提升”。
- 同时，这个桶的 ASV 提升只有 `+0.0022` 左右，也说明 SFT 在口语句上的主要收益更偏向内容微修，而不是音色大幅变化。

`普通句` 桶：

- 这部分几乎已经被 Base 做穿。
- Base 与 `epoch-1` 的 WER 都是 `0.620%`，`90%` 的样本本来就是 `0 WER`，而且 `>=10%` 的样本一条都没有。
- 这说明如果把 benchmark 中的人名地名、成语书面语和口语长尾因素拿掉，当前模型在常规句子上的内容一致性其实已经很强了。

因此，对“是不是因为例子太难”这个问题，可以给出一个更精确的回答：

- `是，但难在长尾词型，而不是难在所有句子都很难。`
- 当前 benchmark 的主要压力，集中在：
  - 人名 / 地名 / 专有词
  - 成语 / 书面长尾词 / 文言感表达
  - 少量口语助词和同音词
- 当前 `scr_sft` 的收益，主要体现为：
  - ASV 在多数桶上小幅提升
  - 口语桶中少数样本的 WER 被修复
- 但它还没有显著提高“专名 + 书面长尾词”上的内容可辨识性，这也是为什么整体 WER 只出现了很小的变化。

**整个流程效果解读**

这次流程最重要的观察，不是“哪个分数最高”，而是“外部 benchmark 最优”和“内部训练期 best checkpoint”并不一致。

- 外部 `seed-tts-eval` 100 条 benchmark 上，三者里最均衡的是 `epoch-1`。
- 但训练内部的 `best_checkpoint.json` 最终记录的是 `epoch-0`，因为内部 best 不是按 WER / SIM 选的。
- `finetuning/scr_sft` 默认只在 `finetuning/exp/output_bznsyp_1p7b_sft_3-25/logs/fixed_eval_set.jsonl` 上做训练期评估，而这个集合只有 `4` 条样本。
- 训练期会对这 `4` 条样本跑 `fixed_eval` 和 `free_run_eval`，并使用自定义 `qc_score` 选模。
- 当前 `best_checkpoint.json` 记录为：
  - `best_epoch = 0`
  - `best_eval_name = free_run_eval`
  - `best_qc_score = 6.6435`
- 相比之下，`epoch-1` 的 `free_run_eval qc_score = 8.8836`，更差，所以没有被内部逻辑选为 best。

这里需要明确指出：内部“稳定性代理指标”与外部“内容 / 音色客观指标”这次并不完全同向。

- 从外部 benchmark 看，`epoch-1` 明显比 `epoch-0` 更合理，因为它同时修复了 WER，并进一步提升了 ASV。
- 从内部 free-run QC 看，`epoch-0` 更像一个“更保守、更不容易被代理指标惩罚”的 checkpoint。
- 这不是代码 bug，而是目标函数与选模函数不同导致的自然结果。

内部 `qc_score` 并不是黑盒，它主要由以下项组成：

- `max_duration_ratio`
- `cap_hit_rate`
- `num_failed_samples`
- `hf_noise_ratio` 超阈值惩罚
- `voiced_f0_delta_p95` 超阈值惩罚
- 峰值 / clipping 惩罚

当前这套设置下，两个 epoch 的内部 4 条样本都被判成了 anomaly：

- `fixed_eval_audio/checkpoint-epoch-0`：`num_failed_samples = 4 / 4`
- `free_run_eval_audio/checkpoint-epoch-0`：`num_failed_samples = 4 / 4`
- `fixed_eval_audio/checkpoint-epoch-1`：`num_failed_samples = 4 / 4`
- `free_run_eval_audio/checkpoint-epoch-1`：`num_failed_samples = 4 / 4`

而且这次内部 `free_run_eval` 的差异，主要并不是由“时长彻底失控”造成的：

- `checkpoint-epoch-0` 与 `checkpoint-epoch-1` 的 `free_run_eval_cap_hit_rate` 都是 `0.0`
- 两者的 `free_run_eval_max_duration_ratio` 都没有超过 `1.0`
- 真正拉开 `qc_score` 的，更像是 `voiced_f0_delta_p95` 与部分噪声相关惩罚项

也就是说，当前内部 QC 更像是一个“相对排序器”，而不是一个可直接解释为“音频质量真的很好 / 很差”的强指标。

**`finetuning/scr_sft` 的合理性**

从代码实现看，这套 SFT 方案是一个非常典型的“保守型单说话人适配”设计，很多地方其实是合理的。

- 它不是全量重训，而是尽量只修改 `talker` 相关模块，目标是降低灾难性遗忘风险。
- `speaker_encoder` 不参与训练，说明训练重点不是重新学习通用 speaker embedding，而是把 base 模型的生成头和说话人槽位适配到目标说话人。
- 训练开始前，会先用训练集第一条样本的 `ref_audio` 初始化 `CUSTOM_SPEAKER_ID = 3000` 对应的 speaker row。
- 训练过程中，又通过 gradient hook 把 `codec_embedding.weight` 的梯度限制在这个自定义 row 上，避免误伤其他说话人槽位。
- 对单说话人场景来说，这种“先初始化 speaker row，再只更新这个 row”的逻辑是合理的，成本低、可控性强。

`staged_stable_sft` 两阶段策略也比较符合经验：

- Stage 1 先训练：
  - `speaker_row`
  - `head`
  - `upper_layers`，即 `talker.model.layers.24-27` 与 `norm`
- Stage 2 再继续放开：
  - `mid_layers`，即 `talker.model.layers.20-23`
  - `talker.code_predictor`

这种安排的直觉很清楚：

- 先改更靠近输出端的部分，把音色和输出头拉向目标说话人
- 再逐步放开中层，提高模型表达自由度，修复 stage1 里可能出现的内容偏移

这和本次实测结果是对得上的：

- `epoch-0` 音色先升，WER 先退
- `epoch-1` 再把内容拉回来，并略微超过 base

KD 的使用同样是合理设计。

- 训练时会加载一个冻结的 base teacher，也就是 `init_model_path` 指向的原始模型。
- loss 不是只有主任务 CE，而是：
  - `main_loss`
  - `0.3 * sub_loss`
  - `main_kd_weight * main_kd_loss`
  - `sub_kd_weight * sub_kd_loss`
- 默认权重是：
  - `main_kd_weight = 0.1`
  - `sub_kd_weight = 0.05`
  - `kd_temperature = 2.0`

这意味着它明确在做一件事：允许模型朝目标说话人方向微调，但又不希望内容生成分布、EOS 行为、时长先验被破坏得太厉害。对小数据单说话人 SFT 来说，这是很稳妥的策略。

导出逻辑也比较工程化：

- 每个 epoch 导出时，会把 checkpoint 的 `config.json` 改写成 `tts_model_type = "custom_voice"`
- 同时把 `talker_config.spk_id` 写成 `{speaker_name: 3000}`
- 推理阶段就可以直接按 `generate_custom_voice()` 使用

这对训练后部署是友好的，因为最终产物本身就是一个可直接推理的 custom voice checkpoint，而不是必须重新拼接一些额外权重。

**`finetuning/scr_sft` 的不足与局限**

虽然这套方案合理，但它的局限也很明确，而且这次结果已经把这些问题暴露得比较充分。

- 内部选模指标和最终外部目标脱节。
  - 训练期不直接看 100 条 benchmark 上的 WER / SIM。
  - 最终导致内部选中了 `epoch-0`，但外部实际更优的是 `epoch-1`。
- 内部评测样本太少。
  - 默认 `fixed_eval_num_samples = 4`。
  - 用 4 条样本来决定早停和最佳 checkpoint，统计波动会非常大，很难稳定代表 100 条测试集。
- `qc_score` 的绝对值解释性偏弱。
  - 当前 `voiced_f0_delta_warn_threshold = 180` 比较严格。
  - 结果是内部 4 条样本在两个 epoch 上都全部被判成 anomaly。
  - 这种情况下，`qc_score` 更像一个相对排序器，而不像一个可靠的“真实主观质量代理”。
- `early_stop_patience = 1` 过于激进。
  - 在只有 4 条评测样本时，任何一次小波动都会很快触发“无提升”判断。
  - 这会进一步放大代理指标与真实外部效果的偏差。
- speaker row 的初始化只用了第一条样本的 `ref_audio`。
  - 这对单说话人场景当然可行，但对鲁棒性和风格覆盖并不理想。
  - 如果第一条参考音频的录制条件、说话风格或情绪比较特殊，初始化会带有偏置。
- 可训练范围整体偏保守。
  - 优点是更稳，不容易把 base 模型训坏。
  - 缺点是适配上限受限，这和本次“提升成立但幅度很小”的结果是一致的。
- Base 与 SFT 的评测推理范式不同。
  - Base 走 voice clone，SFT 走 custom voice。
  - 工程上这当然可接受，因为二者本来就是不同产品形态。
  - 但在分析结果时必须承认：这里面同时混入了“模型能力变化”和“推理范式变化”的因素。
- 当前闭环更像“稳定导出一个可用 custom voice 模型”，而不是“直接面向外部 `seed-tts-eval` 排行榜做指标最优化”。

换句话说，`scr_sft` 现在的优势在于稳健和工程完整性，而不是在于它已经把外部 benchmark 对齐得非常好。

**优缺点总结**

优点：

- 训练策略稳健，比较不容易把 base 模型训坏。
- 工程化程度高，训练、评估、导出、恢复、日志、异常 batch 落盘都比较完整。
- 单说话人适配成本低，speaker row 初始化与梯度 mask 的设计很适合当前场景。
- 导出部署顺滑，checkpoint 直接就是 custom voice 推理形态。
- `epoch-1` 在外部 100 条 benchmark 上同时略优于 base 的 WER 和 ASV，说明 SFT 是有效的。

缺点：

- 提升幅度有限，更像稳健微调成功，而不是显著超越 base。
- 内部最优 checkpoint 不等于外部 benchmark 最优 checkpoint。
- 内部选模代理目标偏弱，且样本数太小。
- 当前早停和 QC 阈值设置较敏感，容易放大小样本波动。
- 对单参考音频初始化依赖较强，风格覆盖能力有限。

综合来看，这次 SFT 可以被评价为：`有效，但收益偏小；更像一次成功的稳健适配，而不是大幅刷新 base 上限。`

如果目标是“稳定拿到一个可用的单说话人 custom voice checkpoint”，当前 `scr_sft` 是合理的。
如果目标是“让外部 `seed-tts-eval` 的 WER / SIM 明显优于 base”，那么下一步更应该优先优化的是训练期评测和选模机制，而不只是继续沿用当前配置多训几个 epoch。

## 10. 设计原则

本工程化重构的目标不是改变训练语义，而是保证：

- 单一正式训练入口
- 单一正式 trainer 编排实现
- 评估、导出、绘图、诊断各自独立
- CLI、目录结构、manifest 字段保持兼容
- 后续新增能力时可以按模块扩展，而不是继续向单个脚本堆功能
