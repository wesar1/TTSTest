import os
import numpy as np
import soundfile as sf
import torch
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

import ChatTTS
from IPython.display import Audio

# 创建输出目录
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

#载入模型
chat = ChatTTS.Chat()
chat.load_models(compile=False)

#批量生成
#texts = ["So we found being competitive and collaborative was a huge way of staying motivated towards our goals, so one person to call when you fall off, one person who gets you back on then one person to actually do the activity with.",]*3 \
#        + ["我觉得像我们这些写程序的人，他，我觉得多多少少可能会对开源有一种情怀在吧我觉得开源是一个很好的形式。现在其实最先进的技术掌握在一些公司的手里的话，就他们并不会轻易的开放给所有的人用。"]*3     
        
#wavs = chat.infer(texts)

#Audio(wavs[0], rate=24_000, autoplay=True)
#Audio(wavs[3], rate=24_000, autoplay=True)

#自定义一些参数，特殊标记来控制文本中的特定语音特征，如语调、笑声和停顿
# use oral_(0-9), laugh_(0-2), break_(0-7) 
params_infer_code = {'prompt':'[speed_5]', 'temperature':.3}
params_refine_text = {'prompt':'[oral_2][laugh_0][break_6]'}

#params_infer_code = {
#  'spk_emb': rand_spk, # add sampled speaker 
#  'temperature': .3, # using custom temperature
#  'top_P': 0.7, # top P decode
#  'top_K': 20, # top K decode
#}

wav = chat.infer('今天天气不错呢！你的离散作业写完了吗？还有微积分和面向对象！', \
    params_refine_text=params_refine_text, params_infer_code=params_infer_code)

# 保存生成的音频
wav = np.array(wav)
sf.write(os.path.join(output_dir, 'output1.wav'), wav.squeeze(), 24_000)
#无法识别“！”“？”，遇到不认识的符号会说出奇怪的话，把!说成了python
#Audio(wav[0], rate=24_000, autoplay=True)



#随机说话人
rand_spk = chat.sample_random_speaker()
params_infer_code = {'spk_emb' : rand_spk, }

wav = chat.infer('大概还有一周就到端午节了，同时也是高考来临的日子，在这里祝大家高考顺利，一顶高粽！', \
    params_refine_text=params_refine_text, params_infer_code=params_infer_code)
    
wav = np.array(wav)
sf.write(os.path.join(output_dir, 'output2.wav'), wav.squeeze(), 24_000)

#Audio(wav[0], rate=24_000, autoplay=True)



###################################
# Sample a speaker from Gaussian.
#加载预训练说话人数据
#std, mean = torch.load('ChatTTS/asset/spk_stat.pt').chunk(2)
#生成嵌入
#rand_spk = torch.randn(768) * std + mean


###################################
# 词级别的调整，通过标记控制语音特征
#text = 'What is [uv_break]your favorite english food?[laugh][lbreak]'
#wav = chat.infer(text, skip_refine_text=True, params_infer_code=params_infer_code)


