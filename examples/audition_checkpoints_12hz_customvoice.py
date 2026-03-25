# coding=utf-8
import argparse
import json
from pathlib import Path

import torch

from qwen_tts import Qwen3TTSModel
from qwen_tts.eval_utils import (
    build_custom_voice_decode_kwargs,
    compute_length_cap,
    summarize_audio_array,
)


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def _resolve_eval_rows(experiment_dir: Path, eval_jsonl: str | None):
    if eval_jsonl:
        eval_path = Path(eval_jsonl)
    else:
        eval_path = experiment_dir / "logs" / "fixed_eval_set.jsonl"
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Fixed eval set not found: {eval_path}. "
            "Pass --eval_jsonl explicitly or run training once with fixed eval enabled."
        )
    return _read_jsonl(eval_path), eval_path


def _resolve_checkpoint_dirs(experiment_dir: Path, checkpoint_names):
    if checkpoint_names:
        checkpoint_dirs = [experiment_dir / name for name in checkpoint_names]
    else:
        checkpoint_dirs = sorted([p for p in experiment_dir.glob("checkpoint-epoch-*") if p.is_dir()])
    missing = [str(p) for p in checkpoint_dirs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Checkpoint(s) not found: {missing}")
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint-epoch-* directory found under {experiment_dir}")
    return checkpoint_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, default="finetuning/exp/output_bznsyp_1p7b_sft-3.21")
    parser.add_argument("--checkpoint_names", nargs="*", default=["checkpoint-epoch-0", "checkpoint-epoch-1"])
    parser.add_argument("--eval_jsonl", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="finetuning/exp/output_bznsyp_1p7b_sft-3.21/listening_tests_3-25")
    parser.add_argument("--speaker", type=str, default="bznsyp_female")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--fixed_eval_length_mode", type=str, choices=["dynamic", "fixed"], default="dynamic")
    parser.add_argument("--fixed_eval_length_multiplier", type=float, default=2.0)
    parser.add_argument("--fixed_eval_max_new_tokens", type=int, default=256)
    parser.add_argument("--fixed_eval_do_sample", action="store_true")
    parser.add_argument("--fixed_eval_language", type=str, default="Chinese")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "listening_tests"
    eval_rows, eval_path = _resolve_eval_rows(experiment_dir, args.eval_jsonl)
    checkpoint_dirs = _resolve_checkpoint_dirs(experiment_dir, args.checkpoint_names)

    dtype = torch.bfloat16
    for checkpoint_dir in checkpoint_dirs:
        import soundfile as sf

        tts = Qwen3TTSModel.from_pretrained(
            str(checkpoint_dir),
            device_map=args.device,
            dtype=dtype,
            attn_implementation=args.attn_implementation,
        )
        checkpoint_output_dir = output_dir / checkpoint_dir.name
        checkpoint_output_dir.mkdir(parents=True, exist_ok=True)

        supported_speakers = tts.get_supported_speakers()
        print(f"[audition] {checkpoint_dir} speakers={supported_speakers}")

        manifest = []
        for idx, row in enumerate(eval_rows):
            sample_id = row.get("sample_id", f"{idx:02d}")
            max_new_tokens = compute_length_cap(
                row,
                length_mode=args.fixed_eval_length_mode,
                length_multiplier=args.fixed_eval_length_multiplier,
                fixed_max_new_tokens=args.fixed_eval_max_new_tokens,
            )
            decode_kwargs = build_custom_voice_decode_kwargs(
                max_new_tokens=max_new_tokens,
                do_sample=args.fixed_eval_do_sample,
            )
            wavs, sample_rate = tts.generate_custom_voice(
                text=row["text"],
                language=row.get("language", args.fixed_eval_language),
                speaker=row.get("speaker", args.speaker),
                instruct=row.get("instruct", ""),
                **decode_kwargs,
            )
            wav_path = checkpoint_output_dir / f"{sample_id}.wav"
            sf.write(wav_path, wavs[0], sample_rate)
            audio_metrics = summarize_audio_array(wavs[0], sample_rate)
            target_seconds = row.get("target_seconds")
            duration_ratio = None
            if target_seconds is not None and float(target_seconds) > 0:
                duration_ratio = round(float(audio_metrics["decoded_seconds"]) / float(target_seconds), 6)
            manifest.append(
                {
                    "index": idx,
                    "sample_id": sample_id,
                    "text": row["text"],
                    "language": row.get("language", args.fixed_eval_language),
                    "instruct": row.get("instruct", ""),
                    "speaker": row.get("speaker", args.speaker),
                    "wav_path": str(wav_path),
                    "sample_rate": sample_rate,
                    "max_new_tokens": max_new_tokens,
                    "decoded_seconds": audio_metrics["decoded_seconds"],
                    "peak": audio_metrics["peak"],
                    "clipped_frac": audio_metrics["clipped_frac"],
                    "target_seconds": target_seconds,
                    "duration_ratio": duration_ratio,
                }
            )

        with (checkpoint_output_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment_dir": str(experiment_dir),
                    "checkpoint_dir": str(checkpoint_dir),
                    "eval_jsonl": str(eval_path),
                    "decode_policy": {
                        "do_sample": bool(args.fixed_eval_do_sample),
                        "subtalker_dosample": bool(args.fixed_eval_do_sample),
                        "fixed_eval_length_mode": args.fixed_eval_length_mode,
                        "fixed_eval_length_multiplier": args.fixed_eval_length_multiplier,
                        "fixed_eval_max_new_tokens": args.fixed_eval_max_new_tokens,
                    },
                    "samples": manifest,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


if __name__ == "__main__":
    main()
