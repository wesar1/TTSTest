from transformers import AutoProcessor, BarkModel
import torch

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 2)

processor = AutoProcessor.from_pretrained("/home/roo/dream/wutr/TTSTest/ckpts/bark")
model = BarkModel.from_pretrained("/home/roo/dream/wutr/TTSTest/ckpts/bark").to(device)

voice_preset = "v2/en_speaker_6"

inputs = processor("Hello, my dog is cute", voice_preset=voice_preset).to(device)

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

import scipy

sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)