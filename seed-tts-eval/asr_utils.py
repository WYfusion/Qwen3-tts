from __future__ import annotations

from pathlib import Path

ZH_WER_MODEL_ALIAS = "paraformer-zh"
ZH_WER_MODELSCOPE_ID = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
EN_WER_MODEL_ID = "openai/whisper-large-v3"


def _looks_like_funasr_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.yaml").exists() and (path / "model.pt").exists()


def resolve_zh_wer_model_dir(modelscope_cache_dir: Path) -> Path | None:
    candidates = [
        modelscope_cache_dir / ZH_WER_MODEL_ALIAS,
        modelscope_cache_dir / "models" / ZH_WER_MODELSCOPE_ID,
        modelscope_cache_dir / ZH_WER_MODELSCOPE_ID,
    ]
    for candidate in candidates:
        if _looks_like_funasr_model_dir(candidate):
            return candidate

    for config_path in modelscope_cache_dir.rglob("config.yaml"):
        candidate = config_path.parent
        if _looks_like_funasr_model_dir(candidate):
            return candidate
    return None


def load_zh_wer_model(*, modelscope_cache_dir: Path):
    try:
        from funasr import AutoModel
    except Exception as exc:
        raise RuntimeError(
            "Chinese WER requires the FunASR Paraformer stack, but importing funasr failed. "
            "Your current environment appears to be Python 3.12 with a hydra/funasr combination "
            "that is not dataclass-compatible. Use a dedicated Python 3.10/3.11 evaluation environment, "
            "or install funasr/hydra versions that support Python 3.12. "
            f"Original import error: {exc}"
        ) from exc

    local_model_dir = resolve_zh_wer_model_dir(modelscope_cache_dir)
    if local_model_dir is not None:
        print(f"Using cached Chinese ASR model: {local_model_dir}", flush=True)
        return AutoModel(model=str(local_model_dir), disable_update=True)

    print(
        "Chinese ASR cache not found locally; attempting to download "
        f"{ZH_WER_MODEL_ALIAS} ({ZH_WER_MODELSCOPE_ID}) via FunASR/ModelScope.",
        flush=True,
    )
    try:
        return AutoModel(model=ZH_WER_MODEL_ALIAS, disable_update=True)
    except AssertionError as exc:
        if "is not registered" not in str(exc):
            raise
        raise RuntimeError(
            "FunASR could not resolve the Chinese ASR alias after download setup. "
            "This usually means the ModelScope download failed before config parsing completed, "
            f"or the local cache is missing. Expected a cached model under {modelscope_cache_dir}. "
            f"If your network cannot reach www.modelscope.cn, pre-download {ZH_WER_MODELSCOPE_ID} "
            "into the ModelScope cache and rerun."
        ) from exc


def resolve_en_wer_model_dir(hf_cache_dir: Path) -> Path | None:
    repo_cache_dir = hf_cache_dir / "models--openai--whisper-large-v3" / "snapshots"
    if not repo_cache_dir.exists():
        return None

    candidates = sorted(path for path in repo_cache_dir.iterdir() if path.is_dir())
    for candidate in reversed(candidates):
        if (candidate / "config.json").exists() and (candidate / "model.safetensors").exists():
            return candidate
    return None
