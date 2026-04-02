from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from asr_utils import EN_WER_MODEL_ID, load_zh_wer_model, resolve_en_wer_model_dir


SIM_UPSTREAM_FILES = {
    "wavlm_large": "wavlm_large.pt",
    "wavlm_base_plus": "wavlm_base_plus.pt",
    "hubert_large": "hubert_large_ll60k.pt",
    "wav2vec2_xlsr": "wav2vec2_xlsr_53_56k.pt",
    "unispeech_sat": "unispeech_sat.pt",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    default_weight_dir = Path(__file__).resolve().parent / "weight"
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--hf_cache_dir", type=str, default=str(default_weight_dir / "huggingface"))
    parser.add_argument("--modelscope_cache_dir", type=str, default=str(default_weight_dir / "modelscope"))
    parser.add_argument("--hf_endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument("--prepare_sim", action="store_true", help="Also prefetch speaker verification upstream checkpoints.")
    parser.add_argument("--sim_model_name", type=str, default="wavlm_large", choices=sorted(SIM_UPSTREAM_FILES))
    parser.add_argument(
        "--sim_finetune_checkpoint",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "weight" / "wavlm_large_finetune.pth"),
        help="Expected path to the speaker verification finetuned checkpoint.",
    )
    return parser


def prepare_sim_assets(*, hf_cache_dir: Path, hf_endpoint: str, sim_model_name: str, sim_finetune_checkpoint: Path) -> None:
    from huggingface_hub import hf_hub_download

    filename = SIM_UPSTREAM_FILES[sim_model_name]
    print(
        f"Preparing SIM upstream checkpoint: model={sim_model_name} repo=s3prl/converted_ckpts "
        f"file={filename} cache={hf_cache_dir} endpoint={hf_endpoint}",
        flush=True,
    )
    upstream_path = hf_hub_download(
        repo_id="s3prl/converted_ckpts",
        filename=filename,
        cache_dir=str(hf_cache_dir),
        endpoint=hf_endpoint,
    )
    print(f"SIM upstream checkpoint ready: {upstream_path}", flush=True)

    sim_finetune_checkpoint = sim_finetune_checkpoint.resolve()
    if sim_finetune_checkpoint.exists():
        print(f"SIM finetuned checkpoint found: {sim_finetune_checkpoint}", flush=True)
    else:
        print(
            f"SIM finetuned checkpoint missing: {sim_finetune_checkpoint}\n"
            f"Please place wavlm_large_finetune.pth there or pass --sim_finetune_checkpoint explicitly.",
            flush=True,
        )


def prepare_zh_wer_assets(*, modelscope_cache_dir: Path) -> None:
    print(f"Preparing ModelScope cache under: {modelscope_cache_dir}", flush=True)
    load_zh_wer_model(modelscope_cache_dir=modelscope_cache_dir)


def prepare_en_wer_assets(*, hf_cache_dir: Path, hf_endpoint: str, device: str) -> None:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    local_model_dir = resolve_en_wer_model_dir(hf_cache_dir)
    model_source = str(local_model_dir) if local_model_dir is not None else EN_WER_MODEL_ID

    if local_model_dir is not None:
        print(f"Using cached English ASR model: {local_model_dir}", flush=True)
        WhisperProcessor.from_pretrained(model_source, local_files_only=True)
        WhisperForConditionalGeneration.from_pretrained(model_source, local_files_only=True).to(device)
        return

    print(f"Using Hugging Face mirror endpoint: {hf_endpoint}")
    WhisperProcessor.from_pretrained(EN_WER_MODEL_ID, cache_dir=str(hf_cache_dir))
    WhisperForConditionalGeneration.from_pretrained(EN_WER_MODEL_ID, cache_dir=str(hf_cache_dir)).to(device)


def main() -> None:
    args = build_parser().parse_args()

    hf_cache_dir = Path(args.hf_cache_dir).resolve()
    modelscope_cache_dir = Path(args.modelscope_cache_dir).resolve()
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    modelscope_cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache_dir / "hub")
    os.environ["MODELSCOPE_CACHE"] = str(modelscope_cache_dir)
    os.environ["HF_ENDPOINT"] = args.hf_endpoint

    print(f"Preparing Hugging Face cache under: {hf_cache_dir}")
    prepare_en_wer_assets(hf_cache_dir=hf_cache_dir, hf_endpoint=args.hf_endpoint, device=args.device)
    prepare_zh_wer_assets(modelscope_cache_dir=modelscope_cache_dir)

    if args.prepare_sim:
        prepare_sim_assets(
            hf_cache_dir=hf_cache_dir,
            hf_endpoint=args.hf_endpoint,
            sim_model_name=args.sim_model_name,
            sim_finetune_checkpoint=Path(args.sim_finetune_checkpoint),
        )


if __name__ == "__main__":
    main()
