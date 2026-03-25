# coding=utf-8

from __future__ import annotations

import json
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
        self.data_list = data_list
        self.processor = processor
        self.lag_num = lag_num
        self.config = config

    def __len__(self):
        return len(self.data_list)

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        audio, sr = librosa.load(x, sr=None, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
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
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _ensure_list(self, x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]

    def _tokenize_texts(self, text) -> torch.Tensor:
        data = self.processor(text=text, return_tensors="pt", padding=True)
        input_ids = data["input_ids"]
        return input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids

    @torch.inference_mode()
    def extract_mels(self, audio, sr):
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
        assert self.lag_num == -1
        item_length = [b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch]
        max_length = max(item_length) + 8
        bsz, total_len = len(batch), max_length
        input_ids = torch.zeros((bsz, total_len, 2), dtype=torch.long)
        codec_ids = torch.zeros((bsz, total_len, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((bsz, total_len), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((bsz, total_len), dtype=torch.bool)
        codec_mask = torch.zeros((bsz, total_len), dtype=torch.bool)
        attention_mask = torch.zeros((bsz, total_len), dtype=torch.long)
        codec_0_labels = torch.full((bsz, total_len), -100, dtype=torch.long)

        for i, data in enumerate(batch):
            text_ids = data["text_ids"]
            audio_codec_0 = data["audio_codes"][:, 0]
            audio_codecs = data["audio_codes"]
            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]

            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i, 7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8 : 8 + text_ids_len - 3, 0] = text_ids[0, 3:]
            input_ids[i, 8 + text_ids_len - 3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8 + text_ids_len - 2 : 8 + text_ids_len + codec_ids_len, 0] = self.config.tts_pad_token_id
            text_embedding_mask[i, : 8 + text_ids_len + codec_ids_len] = True

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

            codec_0_labels[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len] = audio_codec_0
            codec_0_labels[i, 8 + text_ids_len - 1 + codec_ids_len] = self.config.talker_config.codec_eos_token_id
            codec_ids[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, :] = audio_codecs

            codec_embedding_mask[i, 3 : 8 + text_ids_len + codec_ids_len] = True
            codec_embedding_mask[i, 6] = False
            codec_mask[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len] = True
            attention_mask[i, : 8 + text_ids_len + codec_ids_len] = True

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
    path = Path(train_jsonl)
    if len(path.parents) >= 2:
        return path.parents[1].name
    return path.stem


def infer_fixed_eval_source_path(train_jsonl: str) -> Path:
    path = Path(train_jsonl)
    if len(path.parents) >= 2:
        candidate = path.parents[1] / "ft_data" / "test_raw.jsonl"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not infer fixed eval source jsonl from train_jsonl={train_jsonl}. "
        "Pass --fixed_eval_source_jsonl explicitly."
    )


def make_sample_id(idx: int, audio_path: str, used_ids: set[str]) -> str:
    stem = Path(audio_path).stem
    sample_id = stem if stem not in used_ids else f"{idx:02d}_{stem}"
    used_ids.add(sample_id)
    return sample_id


def fixed_eval_rows_are_valid(rows) -> bool:
    if not rows:
        return False
    return all(REQUIRED_FIXED_EVAL_FIELDS.issubset(row.keys()) for row in rows)


def compute_target_code_frames(speech_tokenizer, audio_path: str) -> int:
    enc = speech_tokenizer.encode(audio_path)
    return int(enc.audio_codes[0].shape[0])


def build_fixed_eval_rows(raw_rows, speech_tokenizer, speaker_name: str, default_language: str):
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


def prepare_fixed_eval_set(*, train_jsonl: str, fixed_eval_jsonl: str | None, fixed_eval_source_jsonl: str | None, fixed_eval_num_samples: int, fixed_eval_language: str, speaker_name: str, speech_tokenizer, logs_dir: Path):
    if fixed_eval_num_samples <= 0:
        return [], None, None
    eval_path = Path(fixed_eval_jsonl) if fixed_eval_jsonl else logs_dir / "fixed_eval_set.jsonl"
    source_path = Path(fixed_eval_source_jsonl) if fixed_eval_source_jsonl else infer_fixed_eval_source_path(train_jsonl)

    rebuild = True
    if eval_path.exists():
        rows = read_jsonl(eval_path)
        if fixed_eval_rows_are_valid(rows):
            rebuild = False
    if rebuild:
        source_rows = read_jsonl(source_path)[:fixed_eval_num_samples]
        fixed_rows = build_fixed_eval_rows(
            source_rows,
            speech_tokenizer=speech_tokenizer,
            speaker_name=speaker_name,
            default_language=fixed_eval_language,
        )
        write_jsonl(eval_path, fixed_rows)
    return read_jsonl(eval_path), eval_path, source_path


def dataset_stats(train_data):
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
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    epoch_subset = Subset(dataset, indices)
    return DataLoader(epoch_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

