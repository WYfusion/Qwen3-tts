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

#### 9.1.2 安装 `seed-tts-eval` 依赖

WER 侧依赖：

```bash
pip install -r seed-tts-eval/requirements.txt
```

SIM 侧依赖请参考：

- `seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification/README.md`
- 并提前准备 `wavlm_large_finetune.pth`
- SIM 评估实际依赖两类权重：
  1. `weight/wavlm_large_finetune.pth`
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
其中中文 WER 使用 `funasr` / `paraformer-zh`，如果你当前是 Python 3.12 环境且 `hydra` 导入时报 dataclass mutable default 错误，请改用单独的 Python 3.10/3.11 评估环境。

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
  --sim_finetune_checkpoint weight/wavlm_large_finetune.pth

python seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification/verification_pair_list_v2.py \
  finetuning/exp/output_bznsyp_1p7b_sft_3-25/seed_tts_eval/checkpoint-epoch-1/wav_res_ref_text \
  --model_name wavlm_large \
  --checkpoint weight/wavlm_large_finetune.pth \
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

`verification_pair_list_v2.py` 现在默认会优先复用 `seed-tts-eval/weight/huggingface` 下的 repo-local upstream cache，不再依赖每次运行都在线拉取 `s3prl/s3prl`。如需显式指定本地 upstream ckpt，可追加：

```bash
  --upstream_ckpt /abs/path/to/wavlm_large.pt
```

如果你在一台可联网 Ubuntu 上已经跑通，想把缓存拷到另一台离线或网络受限机器，至少复制以下目录/文件：

```bash
seed-tts-eval/weight/huggingface/
weight/wavlm_large_finetune.pth
```

只要 upstream `wavlm_large.pt` 已在上述 Hugging Face cache 中，SIM 初始化就会优先命中本地缓存；如果缓存缺失且网络不可用，脚本会在模型初始化阶段直接报错退出，而不是在每个样本上重复卡住。

#### 9.1.5 结果解释建议

- `wer.summary.txt` 给出整体 WER
- `sim.summary.txt` 给出平均 speaker similarity
- 当前训练日志里 `checkpoint-epoch-0` 的内部 fixed-eval QC 优于 `checkpoint-epoch-1`，建议两者都跑同一套 `seed-tts-eval` 流程做外部对照
- README 中推荐的流程不依赖 upstream `cal_wer.sh` / `cal_sim.sh`，因为它们偏 Linux / 多卡，且本地 `cal_wer.sh` 末尾汇总路径并不适合直接照搬

## 10. 设计原则

本工程化重构的目标不是改变训练语义，而是保证：

- 单一正式训练入口
- 单一正式 trainer 编排实现
- 评估、导出、绘图、诊断各自独立
- CLI、目录结构、manifest 字段保持兼容
- 后续新增能力时可以按模块扩展，而不是继续向单个脚本堆功能
