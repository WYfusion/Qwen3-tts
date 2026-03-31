# coding=utf-8

from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel
from qwen_tts.eval_utils import summarize_audio_array

from .io_utils import ensure_dir, read_jsonl, safe_round, write_json

DEFAULT_LANGUAGE = "Chinese"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2"
DEFAULT_DTYPE = torch.bfloat16
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_LENGTH_MULTIPLIER = 2.0
CODE_FRAMES_PER_SECOND = 12.5
REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    import soundfile as sf
except ModuleNotFoundError:
    sf = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--eval_jsonl", type=str, required=True)
    parser.add_argument("--speaker_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser


def _resolve_existing_path(path_value: str | Path, *, extra_base_dir: Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        return path.resolve()

    search_roots = [Path.cwd()]
    if extra_base_dir is not None:
        search_roots.append(extra_base_dir)
    search_roots.append(REPO_ROOT)
    for root in search_roots:
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Path not found: {path}")


def _resolve_dataset_media_path(path_value: str | Path, *, eval_jsonl_path: Path) -> Path:
    return _resolve_existing_path(path_value, extra_base_dir=eval_jsonl_path.parent)


def _sync_cuda_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _compute_max_new_tokens(target_seconds: float) -> int:
    dynamic_cap = round(float(target_seconds) * CODE_FRAMES_PER_SECOND * DEFAULT_LENGTH_MULTIPLIER)
    return int(max(2, min(DEFAULT_MAX_NEW_TOKENS, dynamic_cap)))


def _read_audio_info(path: Path) -> dict[str, float | int]:
    if sf is not None:
        info = sf.info(str(path))
        seconds = float(info.frames) / float(info.samplerate)
        return {
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "seconds": seconds,
            "frames": int(info.frames),
        }
    with wave.open(str(path), "rb") as wav_reader:
        frames = int(wav_reader.getnframes())
        sample_rate = int(wav_reader.getframerate())
        channels = int(wav_reader.getnchannels())
    seconds = float(frames) / float(sample_rate)
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "seconds": seconds,
        "frames": frames,
    }


def _write_wav(path: Path, wav: np.ndarray, sample_rate: int) -> None:
    if sf is not None:
        sf.write(path, wav, sample_rate)
        return

    audio = np.asarray(wav)
    if audio.ndim == 1:
        channels = 1
    elif audio.ndim == 2:
        channels = int(audio.shape[1])
    else:
        raise ValueError(f"Unsupported audio shape for wav export: {audio.shape}")
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_writer:
        wav_writer.setnchannels(channels)
        wav_writer.setsampwidth(2)
        wav_writer.setframerate(int(sample_rate))
        wav_writer.writeframes(pcm.tobytes())


def _build_decode_kwargs(target_seconds: float) -> dict[str, int | float | bool]:
    max_new_tokens = _compute_max_new_tokens(target_seconds)
    return {
        "do_sample": False,
        "subtalker_dosample": False,
        "temperature": 1.0,
        "subtalker_temperature": 1.0,
        "top_k": 1,
        "subtalker_top_k": 1,
        "top_p": 1.0,
        "subtalker_top_p": 1.0,
        "max_new_tokens": max_new_tokens,
    }


def _load_tts(checkpoint_dir: Path) -> tuple[Qwen3TTSModel, str | None]:
    try:
        tts = Qwen3TTSModel.from_pretrained(
            str(checkpoint_dir),
            device_map=DEFAULT_DEVICE,
            dtype=DEFAULT_DTYPE,
            attn_implementation=DEFAULT_ATTN_IMPLEMENTATION,
        )
        return tts, DEFAULT_ATTN_IMPLEMENTATION
    except ImportError as exc:
        print(f"flash_attention_2 unavailable, falling back to eager attention: {exc}")
        tts = Qwen3TTSModel.from_pretrained(
            str(checkpoint_dir),
            device_map=DEFAULT_DEVICE,
            dtype=DEFAULT_DTYPE,
            attn_implementation=None,
        )
        return tts, None


def _normalize_eval_rows(eval_jsonl_path: Path):
    raw_rows = read_jsonl(eval_jsonl_path)
    rows = []
    seen_utts = set()
    for index, row in enumerate(raw_rows):
        if "audio" not in row or "text" not in row or "ref_audio" not in row:
            raise KeyError("Each eval row must contain `audio`, `text`, and `ref_audio`.")
        utt = Path(row["audio"]).stem
        if not utt:
            raise ValueError(f"Could not infer utt from audio path: {row['audio']}")
        if utt in seen_utts:
            raise ValueError(f"Duplicate utt detected in eval set: {utt}")
        if "|" in str(row["text"]):
            raise ValueError(f"Text for utt={utt} contains `|`, which breaks seed-tts-eval meta format.")
        seen_utts.add(utt)

        target_audio = _resolve_dataset_media_path(row["audio"], eval_jsonl_path=eval_jsonl_path)
        ref_audio = _resolve_dataset_media_path(row["ref_audio"], eval_jsonl_path=eval_jsonl_path)
        target_meta = _read_audio_info(target_audio)
        rows.append(
            {
                "index": index,
                "utt": utt,
                "text": str(row["text"]),
                "language": str(row.get("language", DEFAULT_LANGUAGE)),
                "instruct": str(row.get("instruct", "")),
                "target_audio": target_audio,
                "ref_audio": ref_audio,
                "target_sample_rate": int(target_meta["sample_rate"]),
                "target_seconds": float(target_meta["seconds"]),
                "target_frames": int(target_meta["frames"]),
                "target_channels": int(target_meta["channels"]),
            }
        )
    if not rows:
        raise ValueError(f"Eval jsonl is empty: {eval_jsonl_path}")
    return rows


def _write_meta(meta_path: Path, rows) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['utt']}|{row['text']}|{row['ref_audio']}\n")


def _write_placeholder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def _render_samples(tts: Qwen3TTSModel, rows, *, speaker_name: str, generated_dir: Path):
    generated_rows = []
    supported_speakers = set(tts.get_supported_speakers())
    if speaker_name not in supported_speakers:
        raise ValueError(
            f"Speaker `{speaker_name}` is not supported by {tts.model.config._name_or_path if hasattr(tts.model.config, '_name_or_path') else 'checkpoint'}: "
            f"{sorted(supported_speakers)}"
        )

    for row in rows:
        decode_kwargs = _build_decode_kwargs(row["target_seconds"])
        _sync_cuda_if_needed()
        start = time.perf_counter()
        wavs, sample_rate = tts.generate_custom_voice(
            text=row["text"],
            language=row["language"],
            speaker=speaker_name,
            instruct=row["instruct"],
            **decode_kwargs,
        )
        _sync_cuda_if_needed()
        wav = wavs[0]
        wav_path = generated_dir / f"{row['utt']}.wav"
        _write_wav(wav_path, wav, sample_rate)
        metrics = summarize_audio_array(wav, sample_rate)
        generated_rows.append(
            {
                "index": row["index"],
                "utt": row["utt"],
                "text": row["text"],
                "language": row["language"],
                "instruct": row["instruct"],
                "speaker": speaker_name,
                "target_audio": row["target_audio"],
                "ref_audio": row["ref_audio"],
                "target_sample_rate": row["target_sample_rate"],
                "target_seconds": safe_round(row["target_seconds"], 6),
                "target_frames": row["target_frames"],
                "generated_wav": wav_path.resolve(),
                "generated_sample_rate": int(sample_rate),
                "generated_seconds": metrics["decoded_seconds"],
                "duration_ratio": safe_round(
                    float(metrics["decoded_seconds"]) / max(float(row["target_seconds"]), 1e-6),
                    6,
                ),
                "peak": metrics["peak"],
                "clipped_frac": metrics["clipped_frac"],
                "hf_noise_ratio": metrics["hf_noise_ratio"],
                "voiced_f0_delta_p95": metrics["voiced_f0_delta_p95"],
                "generation_wall_time": safe_round(time.perf_counter() - start, 4),
                "decode_kwargs": decode_kwargs,
            }
        )
        print(
            f"[seed-tts-eval] {row['utt']} max_new_tokens={decode_kwargs['max_new_tokens']} "
            f"generated_seconds={metrics['decoded_seconds']} peak={metrics['peak']}"
        )
    return generated_rows


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    checkpoint_dir = _resolve_existing_path(args.checkpoint_dir)
    eval_jsonl_path = _resolve_existing_path(args.eval_jsonl)
    output_dir = Path(args.output_dir).resolve()
    generated_dir = ensure_dir(output_dir / "generated")

    eval_rows = _normalize_eval_rows(eval_jsonl_path)
    _write_meta(output_dir / "meta.lst", eval_rows)
    for artifact_name in ("wav_res_ref_text", "wer.raw.txt", "wer.summary.txt", "sim.raw.txt", "sim.summary.txt"):
        _write_placeholder(output_dir / artifact_name)

    tts, attn_implementation = _load_tts(checkpoint_dir)
    generated_rows = _render_samples(
        tts,
        eval_rows,
        speaker_name=args.speaker_name,
        generated_dir=generated_dir,
    )

    write_json(
        output_dir / "manifest.json",
        {
            "checkpoint_dir": checkpoint_dir,
            "eval_jsonl": eval_jsonl_path,
            "output_dir": output_dir,
            "generated_dir": generated_dir,
            "meta_lst": output_dir / "meta.lst",
            "wav_res_ref_text": output_dir / "wav_res_ref_text",
            "wer_raw": output_dir / "wer.raw.txt",
            "wer_summary": output_dir / "wer.summary.txt",
            "sim_raw": output_dir / "sim.raw.txt",
            "sim_summary": output_dir / "sim.summary.txt",
            "num_samples": len(generated_rows),
            "language_default": DEFAULT_LANGUAGE,
            "speaker_name": args.speaker_name,
            "device": DEFAULT_DEVICE,
            "dtype": str(DEFAULT_DTYPE),
            "attn_implementation": attn_implementation,
            "decode_policy": {
                "do_sample": False,
                "subtalker_dosample": False,
                "temperature": 1.0,
                "subtalker_temperature": 1.0,
                "top_k": 1,
                "subtalker_top_k": 1,
                "top_p": 1.0,
                "subtalker_top_p": 1.0,
                "max_new_tokens_rule": "min(256, round(target_seconds * 12.5 * 2.0))",
            },
            "samples": generated_rows,
        },
    )


if __name__ == "__main__":
    main()
