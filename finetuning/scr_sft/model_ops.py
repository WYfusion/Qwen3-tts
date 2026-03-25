# coding=utf-8

from __future__ import annotations

import torch
from accelerate import Accelerator
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

from .constants import CUSTOM_SPEAKER_ID, SPEAKER_SLOT_INDEX


def load_qwen3tts_with_attn_fallback(model_path: str, *, attn_impl: str | None, accelerator: Accelerator):
    load_kwargs = {"torch_dtype": torch.bfloat16}
    if attn_impl is not None:
        load_kwargs["attn_implementation"] = attn_impl
    try:
        return Qwen3TTSModel.from_pretrained(model_path, **load_kwargs)
    except ImportError as exc:
        if attn_impl != "flash_attention_2":
            raise
        accelerator.print(f"flash_attention_2 unavailable, falling back to eager attention: {exc}")
        return Qwen3TTSModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation=None)


def maybe_init_custom_speaker_row(model, dataset, train_data, accelerator: Accelerator):
    if not train_data:
        raise ValueError("train_jsonl is empty.")
    ref_audio = train_data[0].get("ref_audio")
    if not ref_audio:
        raise KeyError("The first training sample does not contain ref_audio.")
    normalized = dataset._normalize_audio_inputs(dataset._ensure_list(ref_audio))
    wav, sr = normalized[0]
    param = next(model.parameters())
    ref_mel = dataset.extract_mels(audio=wav, sr=sr).to(param.device).to(param.dtype)
    with torch.no_grad():
        ref_embedding = model.speaker_encoder(ref_mel)[0]
        embedding_weight = model.talker.model.codec_embedding.weight
        embedding_weight[CUSTOM_SPEAKER_ID].copy_(
            ref_embedding.to(device=embedding_weight.device, dtype=embedding_weight.dtype)
        )
    accelerator.print(f"Initialized custom speaker row {CUSTOM_SPEAKER_ID} from ref_audio={ref_audio}")
    return ref_audio


def build_talker_input_embeddings(model, batch):
    input_ids = batch["input_ids"]
    codec_ids = batch["codec_ids"]
    text_embedding_mask = batch["text_embedding_mask"]
    codec_embedding_mask = batch["codec_embedding_mask"]
    codec_mask = batch["codec_mask"]

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]
    input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
    input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask

    speaker_embedding = model.talker.model.codec_embedding.weight[CUSTOM_SPEAKER_ID].view(1, 1, -1)
    input_codec_embedding[:, SPEAKER_SLOT_INDEX, :] = speaker_embedding.expand(
        input_codec_embedding.shape[0],
        -1,
        -1,
    )[:, 0, :]

    input_embeddings = input_text_embedding + input_codec_embedding
    for i in range(1, 16):
        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
        input_embeddings = input_embeddings + codec_i_embedding
    return input_embeddings


def forward_talker_with_sub_loss(model, batch):
    input_embeddings = build_talker_input_embeddings(model, batch)
    outputs = model.talker(
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=batch["attention_mask"][:, :-1],
        labels=batch["codec_0_labels"][:, 1:],
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[0][-1]
    talker_hidden_states = hidden_states[batch["codec_mask"][:, :-1]]
    talker_codec_ids = batch["codec_ids"][batch["codec_mask"]]
    sub_logits, sub_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
    return outputs, sub_logits, sub_loss

