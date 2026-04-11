# coding=utf-8

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, List, Tuple, Union

import librosa
import numpy as np
import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from torch.utils.data import DataLoader, Dataset, Subset

from .constants import REQUIRED_FIXED_EVAL_FIELDS
from .io_utils import audio_info, read_jsonl, write_jsonl

AudioLike = Union[str, np.ndarray, Tuple[np.ndarray, int]]
MaybeList = Union[Any, List[Any]]


class TTSDataset(Dataset):
    def __init__(self, data_list, processor, config: Qwen3TTSConfig, lag_num=-1):
        """TTS 训练数据集。

        参数:
            data_list: 样本列表，每个样本需包含文本、目标音频编码与参考音频路径等字段。
            processor: 文本处理器/分词器，需支持 text -> input_ids。
            config: Qwen3TTS 模型配置，提供文本与 codec 特殊 token。
            lag_num: 预留参数，当前仅支持 -1。
        """
        self.data_list = data_list
        self.processor = processor
        self.lag_num = lag_num
        self.config = config

    def __len__(self):
        """返回数据集样本数。"""
        return len(self.data_list)

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        """从音频文件读取单声道波形。

        参数:
            x: 音频文件路径。

        返回:
            (audio, sr): float32 波形与采样率。
        """
        audio, sr = librosa.load(x, sr=None, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        """统一参考音频输入格式为 [(waveform, sr), ...]。

        参数:
            audios: 支持单个或列表输入；元素可为文件路径或 (np.ndarray, sr) 二元组。

        返回:
            规范化后的音频列表。
        """
        items = audios if isinstance(audios, list) else [audios]
        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        return out

    def _build_assistant_text(self, text: str) -> str:
        """构造符合 chat 模板的 assistant 文本片段。"""
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _ensure_list(self, x: MaybeList) -> List[Any]:
        """将输入标准化为列表。"""
        return x if isinstance(x, list) else [x]

    def _tokenize_texts(self, text) -> torch.Tensor:
        """对文本进行分词并保证输出为二维张量 [B, T]。这里的分词器采用的是"""
        data = self.processor(text=text, return_tensors="pt", padding=True)
        input_ids = data["input_ids"]
        return input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids

    @torch.inference_mode()
    def extract_mels(self, audio, sr):
        """提取参考音频 Mel 频谱特征。

        参数:
            audio: 1D 波形数组。
            sr: 采样率，必须为 24000。

        返回:
            形状为 [1, T, 128] 的 Mel 特征。
        """
        assert sr == 24000, "Only support 24kHz audio"
        return mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)

    def __getitem__(self, idx):
        """按索引构建单条训练样本。

        关键步骤:
            1. 读取并组织文本与目标 codec。
            2. 加载参考音频并提取 Mel。
            3. 返回训练所需字段与调试元信息。
        """
        item = self.data_list[idx]
        audio_path = item["audio"]
        raw_text = item["text"]
        audio_codes = torch.tensor(item["audio_codes"], dtype=torch.long)
        language = item.get("language", "Auto")
        ref_audio_path = item["ref_audio"]
        text = self._build_assistant_text(raw_text)
        text_ids = self._tokenize_texts(text)
        ref_audio_list = self._ensure_list(ref_audio_path)
        normalized = self._normalize_audio_inputs(ref_audio_list)
        wav, sr = normalized[0]
        ref_mel = self.extract_mels(audio=wav, sr=sr)
        return {
            # 去掉模板尾部 5 个控制 token，避免与后续拼接段重复。
            "text_ids": text_ids[:, :-5],
            "audio_codes": audio_codes,
            "ref_mel": ref_mel,
            "sample_meta": {
                "index": idx,
                "audio": audio_path,
                "text": raw_text,
                "ref_audio": ref_audio_path,
                "language": language,
            },
        }

    def collate_fn(self, batch):
        """将样本列表拼接为模型可直接消费的批次张量。

        该函数是训练数据构造的核心，目标是把每条样本中的:
            1) 文本 token 序列
            2) 语音 codec 序列（16 路码本，主监督为第 0 路）
            3) 参考 Mel 特征
        对齐到统一长度，并同步生成 attention/mask/label。

        参数:
            batch: 由 __getitem__ 返回的样本字典列表。
                每个元素至少包含:
                - text_ids: [1, T_text]
                - audio_codes: [T_codec, 16]
                - ref_mel: [1, T_mel, 128]

        返回:
            dict，关键字段如下:
            - input_ids: [B, T, 2]，双通道输入。通道 0 为文本，通道 1 为 codec-0 控制与内容。
            - ref_mels: [B, T_mel, 128]，参考说话人 Mel 特征。
            - attention_mask: [B, T]，有效时序位置。
            - text_embedding_mask: [B, T, 1]，文本分支的嵌入启用位置。
            - codec_embedding_mask: [B, T, 1]，codec 分支的嵌入启用位置。
            - codec_0_labels: [B, T]，codec-0 的监督标签，忽略位为 -100。
            - codec_ids: [B, T, 16]，完整 16 路码本，用于多码本对齐。
            - codec_mask: [B, T]，真实 codec 内容区间。
            - sample_meta: 原样本元信息列表。
        """
        assert self.lag_num == -1

        # Step 1/8: 计算每条样本的有效时序长度。
        # 这里的长度由“文本长度 + codec 帧数”组成，随后再加固定开销 8（模板控制位）。
        # 后续所有样本都 pad 到同一个 max_length，确保 batch 张量可以堆叠。
        item_length = [b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch]
        max_length = max(item_length) + 8
        bsz, total_len = len(batch), max_length

        # Step 2/8: 初始化批次级容器。
        # input_ids 最后一个维度=2，表示双通道时序输入：
        #   channel 0 -> 文本 token 流
        #   channel 1 -> codec-0 控制 token + codec-0 内容 token
        input_ids = torch.zeros((bsz, total_len, 2), dtype=torch.long)

        # codec_ids 保存完整 16 路码本序列，便于后续模型读取非 0 路码本信息。
        codec_ids = torch.zeros((bsz, total_len, 16), dtype=torch.long)

        # 下面几个 mask 分别控制不同分支的可见性和监督区间：
        # text_embedding_mask: 文本嵌入分支在哪些时刻有效
        # codec_embedding_mask: codec 嵌入分支在哪些时刻有效
        # codec_mask: 真实 codec 内容区间（不含控制位）
        # attention_mask: 总体时序有效区间
        text_embedding_mask = torch.zeros((bsz, total_len), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((bsz, total_len), dtype=torch.bool)
        codec_mask = torch.zeros((bsz, total_len), dtype=torch.bool)
        attention_mask = torch.zeros((bsz, total_len), dtype=torch.long)

        # codec_0_labels 是训练主监督。
        # 仅 codec-0 的目标位置写真实标签；其他位置统一置为 -100 以忽略 loss。
        codec_0_labels = torch.full((bsz, total_len), -100, dtype=torch.long)

        # Step 3/8: 逐样本写入双通道序列与监督信息。
        for i, data in enumerate(batch):
            text_ids = data["text_ids"]
            audio_codec_0 = data["audio_codes"][:, 0]
            audio_codecs = data["audio_codes"]
            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]

            # Step 4/8: 写入文本通道（input_ids[..., 0]）。
            # 时间轴布局（按该实现的固定协议）：
            #   [0:3)             -> text 前 3 个 token（通常是模板头部）
            #   [3:7)             -> tts_pad_token_id（预留控制槽位）
            #   [7]               -> tts_bos_token_id
            #   [8 : 8+T_text-3)  -> text 正文 token（去掉前 3）
            #   [8+T_text-3]      -> tts_eos_token_id
            #   后续一段            -> 文本通道 pad，对齐 codec 区间长度
            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i, 7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8 : 8 + text_ids_len - 3, 0] = text_ids[0, 3:]
            input_ids[i, 8 + text_ids_len - 3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8 + text_ids_len - 2 : 8 + text_ids_len + codec_ids_len, 0] = self.config.tts_pad_token_id

            # 文本嵌入在“文本段 + codec 对齐段”都保持有效，保证双分支时序对齐。
            text_embedding_mask[i, : 8 + text_ids_len + codec_ids_len] = True

            # Step 5/8: 写入 codec-0 通道（input_ids[..., 1]）。
            # 先放 5 个固定控制位：
            #   codec_nothink_id / codec_think_bos_id / codec_think_eos_id / 0 / codec_pad_id
            # 再在正文段填 codec_pad，随后接 codec_bos + codec-0 序列 + codec_eos。
            input_ids[i, 3:8, 1] = torch.tensor(
                [
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                    0,
                    self.config.talker_config.codec_pad_id,
                ]
            )
            input_ids[i, 8 : 8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_ids_len - 2, 1] = self.config.talker_config.codec_bos_id
            input_ids[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, 1] = audio_codec_0
            input_ids[i, 8 + text_ids_len - 1 + codec_ids_len, 1] = self.config.talker_config.codec_eos_token_id

            # Step 6/8: 构造 codec-0 监督标签与完整码本对齐。
            # 监督标签只覆盖真实 codec-0 内容与末尾 eos。
            codec_0_labels[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len] = audio_codec_0
            codec_0_labels[i, 8 + text_ids_len - 1 + codec_ids_len] = self.config.talker_config.codec_eos_token_id

            # 完整 16 路码本在相同时间区间写入，保证与 codec-0 主时轴严格对齐。
            codec_ids[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, :] = audio_codecs

            # Step 7/8: 生成各类 mask。
            # codec_embedding_mask 在 codec 分支总体有效区间为 True，
            # 但位置 6 被明确置 False（该位置是特殊控制槽，不参与 codec 嵌入）。
            codec_embedding_mask[i, 3 : 8 + text_ids_len + codec_ids_len] = True
            codec_embedding_mask[i, 6] = False

            # codec_mask 只标记真实 codec 内容区间（不含 codec_bos/eos）。
            codec_mask[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len] = True

            # attention_mask 标记本样本真实有效长度，超出部分保持 0 作为 padding。
            attention_mask[i, : 8 + text_ids_len + codec_ids_len] = True

        # Step 8/8: 聚合参考 Mel，并返回批次字典。
        # 每条 ref_mel 形状是 [1, T_mel, 128]，按 dim=0 拼接后得到 [B, T_mel, 128]。
        ref_mels = torch.cat([data["ref_mel"] for data in batch], dim=0)
        return {
            "input_ids": input_ids,
            "ref_mels": ref_mels,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
            "sample_meta": [data["sample_meta"] for data in batch],
        }


def infer_dataset_name(train_jsonl: str) -> str:
    """根据训练集 jsonl 路径推断数据集名称。"""
    path = Path(train_jsonl)
    if len(path.parents) >= 2:
        return path.parents[1].name
    return path.stem


def infer_fixed_eval_source_path(train_jsonl: str) -> Path:
    """从训练集路径推断固定评测源文件路径。"""
    path = Path(train_jsonl)
    if len(path.parents) >= 2:
        candidate = path.parents[1] / "ft_data" / "test_raw.jsonl"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not infer fixed eval source jsonl from train_jsonl={train_jsonl}. "
        "Pass --fixed_eval_source_jsonl explicitly."
    )


def infer_benchmark_eval_source_path(train_jsonl: str) -> Path:
    """从训练集路径推断 benchmark 评测源文件路径。"""
    path = Path(train_jsonl)
    candidate = path.with_name("test_with_codes.jsonl")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not infer benchmark eval jsonl from train_jsonl={train_jsonl}. "
        "Pass --benchmark_eval_jsonl explicitly."
    )


def make_sample_id(idx: int, audio_path: str, used_ids: set[str]) -> str:
    """基于音频文件名生成稳定且不重复的 sample_id。"""
    stem = Path(audio_path).stem
    sample_id = stem if stem not in used_ids else f"{idx:02d}_{stem}"
    used_ids.add(sample_id)
    return sample_id


def fixed_eval_rows_are_valid(rows) -> bool:
    """检查固定评测样本字段是否完整。"""
    if not rows:
        return False
    return all(REQUIRED_FIXED_EVAL_FIELDS.issubset(row.keys()) for row in rows)


def compute_target_code_frames(speech_tokenizer, audio_path: str) -> int:
    """计算目标音频编码后的帧数。"""
    enc = speech_tokenizer.encode(audio_path)
    return int(enc.audio_codes[0].shape[0])


def build_fixed_eval_rows(raw_rows, speech_tokenizer, speaker_name: str, default_language: str):
    """将原始评测样本转换为固定评测格式。

    参数:
        raw_rows: 原始样本列表。
        speech_tokenizer: 语音 tokenizer，用于计算目标 codec 帧数。
        speaker_name: 固定写入的说话人名。
        default_language: 缺省语言。

    返回:
        满足固定评测字段规范的样本列表。
    """
    used_ids = set()
    fixed_rows = []
    for idx, row in enumerate(raw_rows):
        info = audio_info(row["audio"])
        fixed_rows.append(
            {
                "sample_id": make_sample_id(idx, row["audio"], used_ids),
                "text": row["text"],
                "language": row.get("language", default_language),
                "instruct": row.get("instruct", ""),
                "speaker": speaker_name,
                "audio": row["audio"],
                "ref_audio": row["ref_audio"],
                "target_sample_rate": info["sample_rate"],
                "target_seconds": round(info["seconds"], 6),
                "target_code_frames": compute_target_code_frames(speech_tokenizer, row["audio"]),
            }
        )
    return fixed_rows


def row_duration_seconds(row: dict) -> float:
    if row.get("target_seconds") is not None:
        return float(row["target_seconds"])
    audio_path = row.get("audio")
    if audio_path:
        return float(audio_info(audio_path)["seconds"])
    audio_codes = row.get("audio_codes")
    if audio_codes:
        return float(len(audio_codes)) / 12.5
    return 0.0


def _stable_row_key(row: dict, idx: int) -> tuple:
    return (
        str(row.get("sample_id", "")),
        str(row.get("audio", "")),
        str(row.get("text", "")),
        int(idx),
    )


def _split_sorted_rows(rows, *, groups: int) -> list[list[tuple[int, dict]]]:
    if groups <= 0:
        return []
    indexed = list(enumerate(rows))
    indexed.sort(key=lambda item: (row_duration_seconds(item[1]), _stable_row_key(item[1], item[0])))
    total = len(indexed)
    base = total // groups
    remainder = total % groups
    buckets = []
    start = 0
    for bucket_idx in range(groups):
        size = base + (1 if bucket_idx < remainder else 0)
        end = start + size
        buckets.append(indexed[start:end])
        start = end
    return buckets


def _allocate_counts(total: int, bucket_sizes: list[int]) -> list[int]:
    if total <= 0 or not bucket_sizes:
        return [0 for _ in bucket_sizes]
    available = sum(bucket_sizes)
    if available <= total:
        return list(bucket_sizes)
    raw = [(float(size) / float(available)) * float(total) if available > 0 else 0.0 for size in bucket_sizes]
    counts = [min(bucket_sizes[idx], int(math.floor(value))) for idx, value in enumerate(raw)]
    remaining = total - sum(counts)
    order = sorted(
        range(len(bucket_sizes)),
        key=lambda idx: (raw[idx] - counts[idx], bucket_sizes[idx], -idx),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for idx in order:
            if counts[idx] >= bucket_sizes[idx]:
                continue
            counts[idx] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return counts


def _pick_evenly_spaced(indexed_rows: list[tuple[int, dict]], count: int) -> list[dict]:
    if count <= 0 or not indexed_rows:
        return []
    if count >= len(indexed_rows):
        return [row for _idx, row in indexed_rows]
    if count == 1:
        return [indexed_rows[len(indexed_rows) // 2][1]]
    positions = [round(i * (len(indexed_rows) - 1) / (count - 1)) for i in range(count)]
    selected = []
    used = set()
    for pos in positions:
        pos = max(0, min(len(indexed_rows) - 1, int(pos)))
        while pos in used and pos + 1 < len(indexed_rows):
            pos += 1
        if pos in used:
            pos = min(idx for idx in range(len(indexed_rows)) if idx not in used)
        used.add(pos)
        selected.append(indexed_rows[pos][1])
    return selected


def duration_stratified_sample(rows, sample_count: int):
    if sample_count <= 0 or not rows:
        return []
    if len(rows) <= sample_count:
        return list(rows)
    buckets = _split_sorted_rows(rows, groups=min(3, len(rows)))
    counts = _allocate_counts(sample_count, [len(bucket) for bucket in buckets])
    picked = []
    for bucket, count in zip(buckets, counts):
        picked.extend(_pick_evenly_spaced(bucket, count))
    indexed_selected = list(enumerate(picked))
    indexed_selected.sort(key=lambda item: _stable_row_key(item[1], item[0]))
    return [row for _idx, row in indexed_selected[:sample_count]]


def prepare_fixed_eval_set(*, train_jsonl: str, fixed_eval_jsonl: str | None, fixed_eval_source_jsonl: str | None, fixed_eval_num_samples: int, fixed_eval_language: str, speaker_name: str, speech_tokenizer, logs_dir: Path):
    """准备固定评测集，必要时自动重建。

    参数:
        train_jsonl: 训练集路径，用于推断默认评测源。
        fixed_eval_jsonl: 固定评测集输出路径；为空则写入 logs_dir。
        fixed_eval_source_jsonl: 固定评测源路径；为空则自动推断。
        fixed_eval_num_samples: 固定评测样本数，<=0 时跳过。
        fixed_eval_language: 缺省语言。
        speaker_name: 说话人名。
        speech_tokenizer: 用于计算目标码帧数。
        logs_dir: 日志目录。

    返回:
        (fixed_rows, eval_path, source_path)
    """
    if fixed_eval_num_samples <= 0:
        return [], None, None
    eval_path = Path(fixed_eval_jsonl) if fixed_eval_jsonl else logs_dir / "fixed_eval_set.jsonl"
    source_path = Path(fixed_eval_source_jsonl) if fixed_eval_source_jsonl else infer_fixed_eval_source_path(train_jsonl)

    # 若已有文件且字段完整则复用；否则按源数据重建。
    rebuild = True
    if eval_path.exists():
        rows = read_jsonl(eval_path)
        if fixed_eval_rows_are_valid(rows) and len(rows) == int(fixed_eval_num_samples):
            rebuild = False
    if rebuild:
        source_rows = duration_stratified_sample(read_jsonl(source_path), fixed_eval_num_samples)
        fixed_rows = build_fixed_eval_rows(
            source_rows,
            speech_tokenizer=speech_tokenizer,
            speaker_name=speaker_name,
            default_language=fixed_eval_language,
        )
        write_jsonl(eval_path, fixed_rows)
    return read_jsonl(eval_path), eval_path, source_path


def dataset_stats(train_data):
    """统计训练数据规模与 codec 长度分布。"""
    code_lengths = [len(item.get("audio_codes", [])) for item in train_data if item.get("audio_codes")]
    ref_audios = {item.get("ref_audio", "") for item in train_data if item.get("ref_audio")}
    return {
        "num_samples": len(train_data),
        "unique_ref_audio_count": len(ref_audios),
        "avg_code_frames": round(sum(code_lengths) / max(1, len(code_lengths)), 4),
        "max_code_frames": max(code_lengths) if code_lengths else 0,
        "min_code_frames": min(code_lengths) if code_lengths else 0,
    }


def build_epoch_dataloader(dataset, batch_size, collate_fn, epoch: int, seed: int):
    """构建按 epoch 可复现打乱顺序的 DataLoader。

    参数:
        dataset: 数据集实例。
        batch_size: 批大小。
        collate_fn: 批处理函数。
        epoch: 当前轮次，用于改变打乱顺序。
        seed: 基础随机种子。

    返回:
        本轮训练使用的 DataLoader。
    """
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    epoch_subset = Subset(dataset, indices)
    return DataLoader(epoch_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
