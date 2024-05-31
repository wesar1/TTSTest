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

description = "A male voice with an Indian accent reads slowly from a book, his words fairly close-sounding and slightly clean. He speaks in a slightly monotone fashion, but his voice is fairly high-pitched, adding a touch of eagerness to his reading."
text_list = [
    "Hello, traveler.",
    "The basic principle of speech synthesis is to convert text into speech waveforms.",
    "The text preprocessing stage includes steps such as text cleaning, word segmentation tagging, in order to extract information acoustic models for subsequent stages. The acoustic stage uses linguistic and acoustic knowledge to convert text into acoustic features."
]

for i, prompt in enumerate(text_list):

    # description = "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. She speaks very slow."

    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids).to(torch.float32)
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(f"role2_test{i}.wav", audio_arr, model.config.sampling_rate)