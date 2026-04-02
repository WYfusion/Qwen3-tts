# part of the code is borrowed from https://github.com/lawlict/ECAPA-TDNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as trans
import os
from pathlib import Path


''' Res2Conv1d + BatchNorm1d + ReLU
'''


class Res2Conv1dReluBn(nn.Module):
    '''
    in_channels == out_channels == channels
    '''

    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)

        return out


''' Conv1d + BatchNorm1d + ReLU
'''


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


''' The SE connection of 1D case.
'''


class SE_Connect(nn.Module):
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)

        return out


''' SE-Res2Block of the ECAPA-TDNN architecture.
'''


# def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
#     return nn.Sequential(
#         Conv1dReluBn(channels, 512, kernel_size=1, stride=1, padding=0),
#         Res2Conv1dReluBn(512, kernel_size, stride, padding, dilation, scale=scale),
#         Conv1dReluBn(512, channels, kernel_size=1, stride=1, padding=0),
#         SE_Connect(channels)
#     )


class SE_Res2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, scale, se_bottleneck_dim):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(out_channels, kernel_size, stride, padding, dilation, scale=scale)
        self.Conv1dReluBn2 = Conv1dReluBn(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)

        return x + residual


''' Attentive weighted mean and standard deviation pooling.
'''


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, attention_channels, kernel_size=1)  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(in_dim, attention_channels, kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)  # equals V and k in the paper

    def forward(self, x):

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x_in))
        # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN(nn.Module):
    def __init__(self, feat_dim=80, channels=512, emb_dim=192, global_context_att=False,
                 feat_type='fbank', sr=16000, feature_selection="hidden_states", update_extract=False, config_path=None,
                 upstream_ckpt=None):
        super().__init__()

        self.feat_type = feat_type
        self.feature_selection = feature_selection
        self.update_extract = update_extract
        self.sr = sr

        if feat_type == "fbank" or feat_type == "mfcc":
            self.update_extract = False

        win_len = int(sr * 0.025)
        hop_len = int(sr * 0.01)

        if feat_type == 'fbank':
            self.feature_extract = trans.MelSpectrogram(sample_rate=sr, n_fft=512, win_length=win_len,
                                                        hop_length=hop_len, f_min=0.0, f_max=sr // 2,
                                                        pad=0, n_mels=feat_dim)
        elif feat_type == 'mfcc':
            melkwargs = {
                'n_fft': 512,
                'win_length': win_len,
                'hop_length': hop_len,
                'f_min': 0.0,
                'f_max': sr // 2,
                'pad': 0
            }
            self.feature_extract = trans.MFCC(sample_rate=sr, n_mfcc=feat_dim, log_mels=False,
                                              melkwargs=melkwargs)
        else:
            if config_path is None:
                print(f"[speaker_verification] loading upstream feature extractor: {feat_type}", flush=True)
                resolved_ckpt = _prefetch_upstream_checkpoint(feat_type, upstream_ckpt=upstream_ckpt)
                print(
                    f"[speaker_verification] loading upstream feature extractor from local ckpt: {resolved_ckpt}",
                    flush=True,
                )
                self.feature_extract = _build_local_upstream_extractor(feat_type, resolved_ckpt)
                print(f"[speaker_verification] upstream feature extractor ready: {feat_type}", flush=True)
            else:
                from .utils import UpstreamExpert

                self.feature_extract = UpstreamExpert(config_path)
            if len(self.feature_extract.model.encoder.layers) == 24 and hasattr(self.feature_extract.model.encoder.layers[23].self_attn, "fp32_attention"):
                self.feature_extract.model.encoder.layers[23].self_attn.fp32_attention = False
            if len(self.feature_extract.model.encoder.layers) == 24 and hasattr(self.feature_extract.model.encoder.layers[11].self_attn, "fp32_attention"):
                self.feature_extract.model.encoder.layers[11].self_attn.fp32_attention = False

            self.feat_num = self.get_feat_num()
            self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))

        if feat_type != 'fbank' and feat_type != 'mfcc':
            freeze_list = ['final_proj', 'label_embs_concat', 'mask_emb', 'project_q', 'quantizer']
            for name, param in self.feature_extract.named_parameters():
                for freeze_val in freeze_list:
                    if freeze_val in name:
                        param.requires_grad = False
                        break

        if not self.update_extract:
            for param in self.feature_extract.parameters():
                param.requires_grad = False

        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        # self.channels = [channels] * 4 + [channels * 3]
        self.channels = [channels] * 4 + [1536]

        self.layer1 = Conv1dReluBn(feat_dim, self.channels[0], kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(self.channels[0], self.channels[1], kernel_size=3, stride=1, padding=2, dilation=2, scale=8, se_bottleneck_dim=128)
        self.layer3 = SE_Res2Block(self.channels[1], self.channels[2], kernel_size=3, stride=1, padding=3, dilation=3, scale=8, se_bottleneck_dim=128)
        self.layer4 = SE_Res2Block(self.channels[2], self.channels[3], kernel_size=3, stride=1, padding=4, dilation=4, scale=8, se_bottleneck_dim=128)

        # self.conv = nn.Conv1d(self.channels[-1], self.channels[-1], kernel_size=1)
        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, self.channels[-1], kernel_size=1)
        self.pooling = AttentiveStatsPool(self.channels[-1], attention_channels=128, global_context_att=global_context_att)
        self.bn = nn.BatchNorm1d(self.channels[-1] * 2)
        self.linear = nn.Linear(self.channels[-1] * 2, emb_dim)


    def get_feat_num(self):
        self.feature_extract.eval()
        wav = [torch.randn(self.sr).to(next(self.feature_extract.parameters()).device)]
        with torch.no_grad():
            features = self.feature_extract(wav)
        select_feature = features[self.feature_selection]
        if isinstance(select_feature, (list, tuple)):
            return len(select_feature)
        else:
            return 1

    def get_feat(self, x):
        if self.update_extract:
            x = self.feature_extract([sample for sample in x])
        else:
            with torch.no_grad():
                if self.feat_type == 'fbank' or self.feat_type == 'mfcc':
                    x = self.feature_extract(x) + 1e-6  # B x feat_dim x time_len
                else:
                    x = self.feature_extract([sample for sample in x])

        if self.feat_type == 'fbank':
            x = x.log()

        if self.feat_type != "fbank" and self.feat_type != "mfcc":
            x = x[self.feature_selection]
            if isinstance(x, (list, tuple)):
                x = torch.stack(x, dim=0)
            else:
                x = x.unsqueeze(0)
            norm_weights = F.softmax(self.feature_weight, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x = (norm_weights * x).sum(dim=0)
            x = torch.transpose(x, 1, 2) + 1e-6

        x = self.instance_norm(x)
        return x

    def forward(self, x):
        x = self.get_feat(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn(self.pooling(out))
        out = self.linear(out)

        return out


def ECAPA_TDNN_SMALL(feat_dim, emb_dim=256, feat_type='fbank', sr=16000, feature_selection="hidden_states", update_extract=False, config_path=None, upstream_ckpt=None):
    return ECAPA_TDNN(feat_dim=feat_dim, channels=512, emb_dim=emb_dim,
                      feat_type=feat_type, sr=sr, feature_selection=feature_selection, update_extract=update_extract, config_path=config_path, upstream_ckpt=upstream_ckpt)


def _local_loader_name(feat_type):
    local_loader_names = {
        "hubert_large_ll60k": "hubert_local",
        "unispeech_sat": "unispeech_sat_local",
        "wav2vec2_xlsr": "wav2vec2_local",
        "wavlm_base_plus": "wavlm_local",
        "wavlm_large": "wavlm_local",
    }
    return local_loader_names.get(feat_type)


def _try_load_with_installed_s3prl(feat_type, resolved_ckpt):
    loader_name = _local_loader_name(feat_type)
    if loader_name is None:
        raise RuntimeError(f"No installed-s3prl local loader mapping for feat_type={feat_type}")
    try:
        import s3prl.hub as s3prl_hub
    except Exception as exc:
        raise RuntimeError(f"installed s3prl import failed: {exc}") from exc
    if not hasattr(s3prl_hub, loader_name):
        raise RuntimeError(f"installed s3prl.hub has no loader named {loader_name}")
    loader = getattr(s3prl_hub, loader_name)
    try:
        return loader(ckpt=resolved_ckpt)
    except TypeError:
        return loader(resolved_ckpt)


def _find_local_torchhub_repo():
    torch_home_env = os.environ.get("TORCH_HOME")
    if not torch_home_env:
        return None
    torch_home = Path(torch_home_env).expanduser()
    hub_dir = torch_home / "hub"
    if not hub_dir.exists():
        return None
    candidates = sorted(hub_dir.glob("s3prl_s3prl_*"))
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "hubconf.py").exists():
            return candidate
    return None


def _try_load_with_local_torchhub(feat_type, resolved_ckpt):
    loader_name = _local_loader_name(feat_type)
    if loader_name is None:
        raise RuntimeError(f"No local torch.hub loader mapping for feat_type={feat_type}")
    repo_dir = _find_local_torchhub_repo()
    if repo_dir is None:
        raise RuntimeError("local torch.hub s3prl repo cache not found under TORCH_HOME/hub")
    try:
        return torch.hub.load(str(repo_dir), loader_name, source="local", ckpt=resolved_ckpt, trust_repo=True)
    except TypeError:
        return torch.hub.load(str(repo_dir), loader_name, source="local", ckpt=resolved_ckpt)


def _build_local_upstream_extractor(feat_type, resolved_ckpt):
    errors = []
    for loader_name, loader in (
        ("installed_s3prl", _try_load_with_installed_s3prl),
        ("local_torchhub", _try_load_with_local_torchhub),
    ):
        try:
            print(
                f"[speaker_verification] trying upstream loader={loader_name} feat_type={feat_type}",
                flush=True,
            )
            feature_extract = loader(feat_type, resolved_ckpt)
            print(
                f"[speaker_verification] upstream loader succeeded: loader={loader_name} feat_type={feat_type}",
                flush=True,
            )
            return feature_extract
        except Exception as exc:
            errors.append(f"{loader_name}: {exc}")
            print(
                f"[speaker_verification] upstream loader failed: loader={loader_name} feat_type={feat_type} error={exc}",
                flush=True,
            )

    try:
        from .utils import UpstreamExpert

        print(
            f"[speaker_verification] falling back to vendored UpstreamExpert for feat_type={feat_type}",
            flush=True,
        )
        return UpstreamExpert(resolved_ckpt)
    except Exception as exc:
        errors.append(f"vendored_upstream_expert: {exc}")
        raise RuntimeError(
            f"All local upstream loading strategies failed for feat_type={feat_type}, ckpt={resolved_ckpt}. "
            f"Attempts: {' | '.join(errors)}"
        ) from exc


def _prefetch_upstream_checkpoint(feat_type, upstream_ckpt=None):
    repo_filenames = {
        "hubert_large_ll60k": "hubert_large_ll60k.pt",
        "unispeech_sat": "unispeech_sat.pt",
        "wav2vec2_xlsr": "wav2vec2_xlsr_53_56k.pt",
        "wavlm_base_plus": "wavlm_base_plus.pt",
        "wavlm_large": "wavlm_large.pt",
    }
    if upstream_ckpt is not None:
        explicit_path = Path(upstream_ckpt).expanduser().resolve()
        if not explicit_path.exists():
            raise FileNotFoundError(
                f"Explicit upstream checkpoint not found for {feat_type}: {explicit_path}"
            )
        print(
            f"[speaker_verification] using explicit upstream checkpoint: {explicit_path}",
            flush=True,
        )
        return str(explicit_path)
    filename = repo_filenames.get(feat_type)
    if filename is None:
        return None

    cache_dir = os.environ.get("HF_HOME")
    endpoint = os.environ.get("HF_ENDPOINT")

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import huggingface_hub while resolving upstream checkpoint for {feat_type}: {exc}"
        ) from exc

    try:
        local_path = hf_hub_download(
            repo_id="s3prl/converted_ckpts",
            filename=filename,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        print(f"[speaker_verification] upstream checkpoint cache hit: {local_path}", flush=True)
        return local_path
    except LocalEntryNotFoundError:
        print(
            f"[speaker_verification] upstream checkpoint cache miss: repo=s3prl/converted_ckpts "
            f"file={filename} cache_dir={cache_dir} endpoint={endpoint}",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[speaker_verification] upstream checkpoint cache probe failed: {exc}",
            flush=True,
        )
    try:
        print(
            f"[speaker_verification] downloading upstream checkpoint via HF mirror: "
            f"repo=s3prl/converted_ckpts file={filename} endpoint={endpoint}",
            flush=True,
        )
        local_path = hf_hub_download(
            repo_id="s3prl/converted_ckpts",
            filename=filename,
            cache_dir=cache_dir,
            endpoint=endpoint,
        )
        print(f"[speaker_verification] upstream checkpoint cached: {local_path}", flush=True)
        return local_path
    except Exception as exc:
        raise RuntimeError(
            f"Unable to resolve upstream checkpoint for {feat_type}. "
            f"Expected cache_dir={cache_dir}, endpoint={endpoint}, filename={filename}. "
            f"You can pre-download it into the repo-local Hugging Face cache or pass --upstream_ckpt. "
            f"Original error: {exc}"
        ) from exc

if __name__ == '__main__':
    x = torch.zeros(2, 32000)
    model = ECAPA_TDNN_SMALL(feat_dim=768, emb_dim=256, feat_type='hubert_base', feature_selection="hidden_states",
                              update_extract=False)

    out = model(x)
    # print(model)
    print(out.shape)

