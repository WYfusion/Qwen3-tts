import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel
from qwen_tts.eval_utils import build_custom_voice_decode_kwargs, summarize_audio_array


DEFAULT_CHECKPOINT = "./finetuning/exp/output_bznsyp_1p7b_sft_3-25/checkpoint-epoch-1"
DEFAULT_OUTPUT_DIR = "examples/wav_3-26"
DEFAULT_SPEAKER = "bznsyp_female"
DEFAULT_CN_TEXT = "其实我真的有发现，我是一个特别善于观察别人情绪的人。"
DEFAULT_CN_STRESS_INSTRUCT = "用特别愤怒的语气说"
DEFAULT_EN_TEXT = "She said she would be here by noon."
DEFAULT_EN_INSTRUCT = "Very happy."


def _sync_cuda_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_case(model: Qwen3TTSModel, output_dir: Path, case_name: str, generate_kwargs: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    _sync_cuda_if_needed()
    start = time.time()
    wavs, sample_rate = model.generate_custom_voice(**generate_kwargs)
    _sync_cuda_if_needed()
    elapsed_seconds = round(time.time() - start, 3)

    records = []
    for index, wav in enumerate(wavs):
        wav_path = output_dir / f"{case_name}_{index}.wav"
        sf.write(wav_path, wav, sample_rate)
        audio_metrics = summarize_audio_array(wav, sample_rate)
        scalar_kwargs = {}
        for key in ("text", "language", "speaker", "instruct"):
            value = generate_kwargs.get(key)
            if isinstance(value, list):
                scalar_kwargs[key] = value[index]
            else:
                scalar_kwargs[key] = value
        record = {
            "case_name": case_name,
            "index": index,
            "wav_path": str(wav_path),
            "elapsed_seconds": elapsed_seconds,
            **scalar_kwargs,
            "decode_kwargs": {
                "do_sample": bool(generate_kwargs.get("do_sample", False)),
                "subtalker_dosample": bool(generate_kwargs.get("subtalker_dosample", False)),
                "max_new_tokens": int(generate_kwargs["max_new_tokens"]),
            },
            **audio_metrics,
        }
        records.append(record)
        print(
            f"[{case_name}:{index}] elapsed={elapsed_seconds:.3f}s "
            f"decoded={audio_metrics['decoded_seconds']:.3f}s "
            f"peak={audio_metrics['peak']:.4f} "
            f"clipped_frac={audio_metrics['clipped_frac']:.6f}"
        )
    return records


def _write_manifest(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--speaker", type=str, default=DEFAULT_SPEAKER)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--mode", type=str, choices=["stable", "stress", "all"], default="all")
    parser.add_argument("--stable_max_new_tokens", type=int, default=96)
    parser.add_argument("--stress_max_new_tokens", type=int, default=160)
    args = parser.parse_args()

    model = Qwen3TTSModel.from_pretrained(
        args.checkpoint,
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    print(model.get_supported_speakers())

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifests = {}

    if args.mode in {"stable", "all"}:
        stable_kwargs = build_custom_voice_decode_kwargs(
            max_new_tokens=args.stable_max_new_tokens,
            do_sample=False,
        )
        stable_records = _run_case(
            model,
            output_root / "stable_demo",
            "stable_demo_cn_plain",
            {
                "text": DEFAULT_CN_TEXT,
                "language": "Chinese",
                "speaker": args.speaker,
                "instruct": "",
                **stable_kwargs,
            },
        )
        manifests["stable_demo"] = {
            "checkpoint": args.checkpoint,
            "speaker": args.speaker,
            "mode": "stable_demo",
            "decode_kwargs": stable_kwargs,
            "samples": stable_records,
        }
        _write_manifest(output_root / "stable_demo" / "manifest.json", manifests["stable_demo"])

    if args.mode in {"stress", "all"}:
        stress_kwargs = build_custom_voice_decode_kwargs(
            max_new_tokens=args.stress_max_new_tokens,
            do_sample=False,
        )
        stress_records = []
        stress_records.extend(
            _run_case(
                model,
                output_root / "stress_demo",
                "stress_demo_cn_emotion",
                {
                    "text": DEFAULT_CN_TEXT,
                    "language": "Chinese",
                    "speaker": args.speaker,
                    "instruct": DEFAULT_CN_STRESS_INSTRUCT,
                    **stress_kwargs,
                },
            )
        )
        stress_records.extend(
            _run_case(
                model,
                output_root / "stress_demo",
                "stress_demo_en_happy",
                {
                    "text": DEFAULT_EN_TEXT,
                    "language": "English",
                    "speaker": args.speaker,
                    "instruct": DEFAULT_EN_INSTRUCT,
                    **stress_kwargs,
                },
            )
        )
        manifests["stress_demo"] = {
            "checkpoint": args.checkpoint,
            "speaker": args.speaker,
            "mode": "stress_demo",
            "decode_kwargs": stress_kwargs,
            "samples": stress_records,
        }
        _write_manifest(output_root / "stress_demo" / "manifest.json", manifests["stress_demo"])

    if args.mode == "all":
        _write_manifest(
            output_root / "manifest.json",
            {
                "checkpoint": args.checkpoint,
                "speaker": args.speaker,
                "modes": manifests,
            },
        )


if __name__ == "__main__":
    main()
