import soundfile as sf
import torch
import fire
import torch.nn.functional as F
import torchaudio
import os
import sys
import types
from pathlib import Path
from torchaudio.transforms import Resample
import librosa

MODEL_LIST = ['ecapa_tdnn', 'hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', "wavlm_base_plus", "wavlm_large"]
WEIGHT_ROOT = Path(__file__).resolve().parents[4] / "weight"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def _setup_local_runtime_cache():
    torch_home = WEIGHT_ROOT / "torch"
    nltk_data = WEIGHT_ROOT / "nltk_data"
    hf_home = WEIGHT_ROOT / "huggingface"
    hf_hub_cache = hf_home / "hub"
    transformers_cache = hf_home / "transformers"
    torch_home.mkdir(parents=True, exist_ok=True)
    nltk_data.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_hub_cache.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)
    # Force the whole speaker-verification stack onto repo-local caches and mirror.
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ["NLTK_DATA"] = str(nltk_data)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    os.environ["HF_ENDPOINT"] = DEFAULT_HF_ENDPOINT


def _patch_torchaudio_compat():
    # s3prl / older upstream code may still call these global backend helpers,
    # but recent torchaudio versions removed them.
    if not hasattr(torchaudio, "set_audio_backend"):
        def _set_audio_backend(_backend):
            return None
        torchaudio.set_audio_backend = _set_audio_backend
    if not hasattr(torchaudio, "get_audio_backend"):
        def _get_audio_backend():
            return None
        torchaudio.get_audio_backend = _get_audio_backend
    if not hasattr(torchaudio, "list_audio_backends"):
        def _list_audio_backends():
            return []
        torchaudio.list_audio_backends = _list_audio_backends
    if "torchaudio.sox_effects" not in sys.modules:
        sox_effects = types.ModuleType("torchaudio.sox_effects")

        def _apply_effects_tensor(waveform, sample_rate, effects):
            output = waveform
            for effect in effects:
                if not effect:
                    continue
                effect_name = effect[0]
                if effect_name == "channels" and len(effect) > 1 and str(effect[1]) == "1":
                    if output.dim() == 1:
                        output = output.unsqueeze(0)
                    if output.shape[0] > 1:
                        output = output.mean(dim=0, keepdim=True)
            return output, sample_rate

        sox_effects.apply_effects_tensor = _apply_effects_tensor
        sys.modules["torchaudio.sox_effects"] = sox_effects
        torchaudio.sox_effects = sox_effects


_setup_local_runtime_cache()
_patch_torchaudio_compat()
from models.ecapa_tdnn import ECAPA_TDNN_SMALL


def init_model(model_name, checkpoint=None, upstream_ckpt=None):
    print(
        f"[speaker_verification] init_model start: model_name={model_name} "
        f"checkpoint={checkpoint} upstream_ckpt={upstream_ckpt}",
        flush=True,
    )
    if model_name == 'unispeech_sat':
        config_path = 'config/unispeech_sat.th'
        model = ECAPA_TDNN_SMALL(
            feat_dim=1024,
            feat_type='unispeech_sat',
            config_path=config_path,
            upstream_ckpt=upstream_ckpt,
        )
    elif model_name == 'wavlm_base_plus':
        config_path = None
        model = ECAPA_TDNN_SMALL(
            feat_dim=768,
            feat_type='wavlm_base_plus',
            config_path=config_path,
            upstream_ckpt=upstream_ckpt,
        )
    elif model_name == 'wavlm_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(
            feat_dim=1024,
            feat_type='wavlm_large',
            config_path=config_path,
            upstream_ckpt=upstream_ckpt,
        )
    elif model_name == 'hubert_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(
            feat_dim=1024,
            feat_type='hubert_large_ll60k',
            config_path=config_path,
            upstream_ckpt=upstream_ckpt,
        )
    elif model_name == 'wav2vec2_xlsr':
        config_path = None
        model = ECAPA_TDNN_SMALL(
            feat_dim=1024,
            feat_type='wav2vec2_xlsr',
            config_path=config_path,
            upstream_ckpt=upstream_ckpt,
        )
    else:
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type='fbank')


    if checkpoint is not None:
        print(f"[speaker_verification] loading checkpoint: {checkpoint}", flush=True)
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
    print(f"[speaker_verification] init_model done: model_name={model_name}", flush=True)
    return model


def verification(model_name,  wav1, wav2, use_gpu=True, checkpoint=None, wav1_start_sr=0, wav2_start_sr=0, wav1_end_sr=-1, wav2_end_sr=-1, model=None, wav2_cut_wav1=False, device="cuda:0", upstream_ckpt=None):

    assert model_name in MODEL_LIST, 'The model_name should be in {}'.format(MODEL_LIST)
    first_init = model is None
    model = init_model(model_name, checkpoint, upstream_ckpt=upstream_ckpt) if model is None else model
    if first_init:
        print(f"[speaker_verification] first verification pass on device={device}", flush=True)

    wav1, sr1 = librosa.load(wav1, sr=None, mono=False)

    # wav1, sr1 = sf.read(wav1)
    if len(wav1.shape) == 2:
        wav1 = wav1[:,0]
    # wav2, sr2 = sf.read(wav2)
    wav2, sr2 = librosa.load(wav2, sr=None, mono=False)
    if len(wav2.shape) == 2:
        wav2 = wav2[0,:]    # wav2.shape: [channels, T]

    wav1 = torch.from_numpy(wav1).unsqueeze(0).float()
    wav2 = torch.from_numpy(wav2).unsqueeze(0).float()
    resample1 = Resample(orig_freq=sr1, new_freq=16000)
    resample2 = Resample(orig_freq=sr2, new_freq=16000)
    wav1 = resample1(wav1)
    wav2 = resample2(wav2)
    # print(f'origin wav1 sr: {wav1.shape}, wav2 sr: {wav2.shape}')
    if wav2_cut_wav1:
        wav2 = wav2[...,wav1.shape[-1]:]
    else:
        wav1 = wav1[...,wav1_start_sr:wav1_end_sr if wav1_end_sr > 0 else wav1.shape[-1]]
        wav2 = wav2[...,wav2_start_sr:wav2_end_sr if wav2_end_sr > 0 else wav2.shape[-1]]
    # print(f'cutted wav1 sr: {wav1.shape}, wav2 sr: {wav2.shape}')

    if use_gpu:
        model = model.cuda(device)
        wav1 = wav1.cuda(device)
        wav2 = wav2.cuda(device)

    model.eval()
    with torch.no_grad():
        emb1 = model(wav1)
        emb2 = model(wav2)

    sim = F.cosine_similarity(emb1, emb2)
    # print("The similarity score between two audios is {:.4f} (-1.0, 1.0).".format(sim[0].item()))
    return sim, model


def extract_embedding(model_name,  wav1, use_gpu=True, checkpoint=None, wav1_start_sr=0, wav1_end_sr=-1, model=None, device="cuda:0", upstream_ckpt=None):

    assert model_name in MODEL_LIST, 'The model_name should be in {}'.format(MODEL_LIST)
    model = init_model(model_name, checkpoint, upstream_ckpt=upstream_ckpt) if model is None else model

    wav1, sr1 = sf.read(wav1)
    wav1 = torch.from_numpy(wav1).unsqueeze(0).float()
    resample1 = Resample(orig_freq=sr1, new_freq=16000)
    wav1 = resample1(wav1)
    # print(f'origin wav1 sr: {wav1.shape}, wav2 sr: {wav2.shape}')
    wav1 = wav1[...,wav1_start_sr:wav1_end_sr if wav1_end_sr > 0 else wav1.shape[-1]]
    if use_gpu:
        model = model.cuda(device)
        wav1 = wav1.cuda(device)

    model.eval()
    with torch.no_grad():
        emb1 = model(wav1)
    # print("The similarity score between two audios is {:.4f} (-1.0, 1.0).".format(sim[0].item()))
    return emb1, model

if __name__ == "__main__":
    fire.Fire(verification)

