from transformers import AutoProcessor, BarkModel
import torch
import scipy

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 2)

processor = AutoProcessor.from_pretrained("/home/roo/dream/wutr/TTSTest/ckpts/bark")
model = BarkModel.from_pretrained("/home/roo/dream/wutr/TTSTest/ckpts/bark").to(device)

en_text_list = [
    "Hello, traveler.",
    "The basic principle of speech synthesis is to convert text into speech waveforms.",
    "The text preprocessing stage includes steps such as text cleaning, word segmentation tagging, in order to extract information acoustic models for subsequent stages. The acoustic stage uses linguistic and acoustic knowledge to convert text into acoustic features."
]

zh_text_list = [
    "你好，旅行者。",
    "语音合成的基本原理是将文本转换为语音波形。",
    "文本预处理阶段包括文本清洗、分词、词性标注等步骤，以便提取出用于后续阶段的信息。声学模型阶段利用语言学和声学知识，将文本转换为声学特征。",
]
voice_preset = "v2/en_speaker_4"

for i, text in enumerate(en_text_list):
    inputs = processor(text, voice_preset=voice_preset).to(device)

    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()



    sample_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write(f"en_speaker_4_test{i}.wav", rate=sample_rate, data=audio_array)