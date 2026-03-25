import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "./finetuning/exp/output_bznsyp_1p7b_sft-3.21/checkpoint-epoch-1",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
print(model.get_supported_speakers())

# single inference
wavs, sr = model.generate_custom_voice(
    text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    language="Chinese", # Pass `Auto` (or omit) for auto language adaptive; if the target language is known, set it explicitly.
    speaker="bznsyp_female",
    instruct="用特别愤怒的语气说", # Omit if not needed.
)
sf.write("examples/wav_3-23/output_custom_voice.wav", wavs[0], sr)

# batch inference
wavs, sr = model.generate_custom_voice(
    text=[
        "其实我真的有发现，我是一个特别善于观察别人情绪的人。", 
        "She said she would be here by noon."
    ],
    language=["Chinese", "English"],
    speaker=["bznsyp_female", "bznsyp_female"],
    instruct=["", "Very happy."]
)
sf.write("examples/wav_3-23/output_custom_voice_1.wav", wavs[0], sr)
sf.write("examples/wav_3-23/output_custom_voice_2.wav", wavs[1], sr)
