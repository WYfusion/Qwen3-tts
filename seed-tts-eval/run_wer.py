from __future__ import annotations

import argparse
import os
import string
from pathlib import Path
import sys

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from asr_utils import EN_WER_MODEL_ID, load_zh_wer_model, resolve_en_wer_model_dir

DEFAULT_WEIGHT_DIR = Path(__file__).resolve().parent / "weight"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_res_text_path", type=str)
    parser.add_argument("res_path", type=str)
    parser.add_argument("lang", choices=["zh", "en"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--hf_cache_dir", type=str, default=str(DEFAULT_WEIGHT_DIR / "huggingface"))
    parser.add_argument("--modelscope_cache_dir", type=str, default=str(DEFAULT_WEIGHT_DIR / "modelscope"))
    parser.add_argument("--hf_endpoint", type=str, default="https://hf-mirror.com")
    return parser


def setup_cache_env(*, hf_cache_dir: Path, modelscope_cache_dir: Path, hf_endpoint: str) -> None:
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    modelscope_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache_dir / "hub")
    os.environ["MODELSCOPE_CACHE"] = str(modelscope_cache_dir)
    os.environ["HF_ENDPOINT"] = hf_endpoint


def load_en_model(*, device: str, hf_cache_dir: Path):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    local_model_dir = resolve_en_wer_model_dir(hf_cache_dir)
    if local_model_dir is not None:
        processor = WhisperProcessor.from_pretrained(str(local_model_dir), local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(str(local_model_dir), local_files_only=True).to(device)
        return processor, model

    processor = WhisperProcessor.from_pretrained(EN_WER_MODEL_ID, cache_dir=str(hf_cache_dir))
    model = WhisperForConditionalGeneration.from_pretrained(EN_WER_MODEL_ID, cache_dir=str(hf_cache_dir)).to(device)
    return processor, model


def load_zh_model(*, modelscope_cache_dir: Path):
    return load_zh_wer_model(modelscope_cache_dir=modelscope_cache_dir)


def process_one(*, hypo: str, truth: str, lang: str):
    from zhon.hanzi import punctuation

    punctuation_all = punctuation + string.punctuation

    raw_truth = truth
    raw_hypo = hypo

    for x in punctuation_all:
        if x == "'":
            continue
        truth = truth.replace(x, "")
        hypo = hypo.replace(x, "")

    truth = truth.replace("  ", " ")
    hypo = hypo.replace("  ", " ")

    if lang == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    elif lang == "en":
        truth = truth.lower()
        hypo = hypo.lower()
    else:
        raise NotImplementedError

    ref_list = truth.split(" ")
    ref_len = max(1, len(ref_list))

    try:
        from jiwer import process_words

        measures = process_words(truth, hypo)
        wer = float(measures.wer)
        subs = float(measures.substitutions) / ref_len
        dele = float(measures.deletions) / ref_len
        inse = float(measures.insertions) / ref_len
    except ImportError:
        from jiwer import compute_measures

        measures = compute_measures(truth, hypo)
        wer = float(measures["wer"])
        subs = float(measures["substitutions"]) / ref_len
        dele = float(measures["deletions"]) / ref_len
        inse = float(measures["insertions"]) / ref_len
    return raw_truth, raw_hypo, wer, subs, dele, inse


def run_asr(*, wav_res_text_path: Path, res_path: Path, lang: str, device: str, hf_cache_dir: Path, modelscope_cache_dir: Path):
    import scipy
    import soundfile as sf
    import zhconv

    if lang == "en":
        processor, model = load_en_model(device=device, hf_cache_dir=hf_cache_dir)
    elif lang == "zh":
        model = load_zh_model(modelscope_cache_dir=modelscope_cache_dir)
    else:
        raise NotImplementedError

    params = []
    for line in wav_res_text_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) == 2:
            wav_res_path, text_ref = parts
        elif len(parts) == 3:
            wav_res_path, _, text_ref = parts
        elif len(parts) == 4:
            wav_res_path, _, text_ref, _ = parts
        else:
            raise NotImplementedError

        if not os.path.exists(wav_res_path):
            continue
        params.append((wav_res_path, text_ref))

    res_path.parent.mkdir(parents=True, exist_ok=True)
    with res_path.open("w", encoding="utf-8") as fout:
        for wav_res_path, text_ref in tqdm(params):
            if lang == "en":
                wav, sr = sf.read(wav_res_path)
                if sr != 16000:
                    wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
                input_features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features
                input_features = input_features.to(device)
                forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
                predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            else:
                res = model.generate(input=wav_res_path, batch_size_s=300)
                transcription = zhconv.convert(res[0]["text"], "zh-cn")

            raw_truth, raw_hypo, wer, subs, dele, inse = process_one(
                hypo=transcription,
                truth=text_ref,
                lang=lang,
            )
            fout.write(f"{wav_res_path}\t{wer}\t{raw_truth}\t{raw_hypo}\t{inse}\t{dele}\t{subs}\n")
            fout.flush()


def main() -> None:
    args = build_parser().parse_args()
    hf_cache_dir = Path(args.hf_cache_dir).resolve()
    modelscope_cache_dir = Path(args.modelscope_cache_dir).resolve()
    setup_cache_env(
        hf_cache_dir=hf_cache_dir,
        modelscope_cache_dir=modelscope_cache_dir,
        hf_endpoint=args.hf_endpoint,
    )
    print(f"Using Hugging Face cache: {hf_cache_dir}")
    print(f"Using ModelScope cache: {modelscope_cache_dir}")
    print(f"Using Hugging Face mirror endpoint: {args.hf_endpoint}")
    run_asr(
        wav_res_text_path=Path(args.wav_res_text_path).resolve(),
        res_path=Path(args.res_path).resolve(),
        lang=args.lang,
        device=args.device,
        hf_cache_dir=hf_cache_dir,
        modelscope_cache_dir=modelscope_cache_dir,
    )


if __name__ == "__main__":
    main()
