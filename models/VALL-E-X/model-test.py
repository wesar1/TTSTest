from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

# download and load all models
preload_models()

# generate audio from text
# text_prompt = """
# Hello, my name is Nose. And uh, and I like hamburger. Hahaha... But I also have other interests such as playing tactic toast.
# """

text_list = [
    "你好，旅行者。",
    "Hello, traveler.",
    "语音合成的基本原理是将文本转换为语音波形。",
    "The basic principle of speech synthesis is to convert text into speech waveforms.",
    "文本预处理阶段包括文本清洗、分词、词性标注等步骤，以便提取出用于后续阶段的信息。声学模型阶段利用语言学和声学知识，将文本转换为声学特征。",
    "The text preprocessing stage includes steps such as text cleaning, word segmentation tagging, in order to extract information acoustic models for subsequent stages. The acoustic stage uses linguistic and acoustic knowledge to convert text into acoustic features."
]

for i, text_prompt in enumerate(text_list): 

    audio_array = generate_audio(text_prompt, prompt='babara')

    # save audio to disk
    write_wav(f"babara{i}.wav", SAMPLE_RATE, audio_array)

# play text in notebook
# Audio(audio_array, rate=SAMPLE_RATE)