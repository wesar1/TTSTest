from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

# download and load all models
preload_models()

# generate audio from text
# text_prompt = """
# Hello, my name is Nose. And uh, and I like hamburger. Hahaha... But I also have other interests such as playing tactic toast.
# """
text_prompt = """
虽然不知道你要找的是不是风之神，但…
我先带你来风神的领地，也是有理由的喔。
"""
audio_array = generate_audio(text_prompt, prompt='paimon')

# save audio to disk
write_wav("vallex_generation.wav", SAMPLE_RATE, audio_array)

# play text in notebook
# Audio(audio_array, rate=SAMPLE_RATE)