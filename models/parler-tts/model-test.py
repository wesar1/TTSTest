from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import torch

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:3"
if torch.backends.mps.is_available():
    device = "mps"
# if torch.xpu.is_available():
#     device = "xpu"
torch_dtype = torch.float32

model = ParlerTTSForConditionalGeneration.from_pretrained("/home/roo/dream/wutr/TTSTest/ckpts/parler_tts_mini_v0.1").to(device, dtype=torch_dtype)
tokenizer = AutoTokenizer.from_pretrained("/home/roo/dream/wutr/TTSTest/ckpts/parler_tts_mini_v0.1")

prompt = "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. She speaks very fast."
description = "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. She speaks very slow."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids).to(torch.float32)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)