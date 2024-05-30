# 微调

显然, 当你打开这个页面的时候, 你已经对预训练模型 few-shot 的效果不算满意. 你想要微调一个模型, 使得它在你的数据集上表现更好.  

`Fish Speech` 由三个模块组成: `VQGAN`,`LLAMA`, 以及 `VITS Decoder`. 

!!! info 
    你应该先进行如下测试来判断你是否需要微调 `VITS Decoder`
    ```bash
    python tools/vqgan/inference.py -i test.wav
    python tools/vits_decoder/inference.py \
        -ckpt checkpoints/vits_decoder_v1.1.ckpt \
        -i fake.npy -r test.wav \
        --text "合成文本"
    ```
    该测试会生成一个 `fake.wav` 文件, 如果该文件的音色和说话人的音色不同, 或者质量不高, 你需要微调 `VITS Decoder`.

    相应的, 你可以参考 [推理](inference.md) 来运行 `generate.py`, 判断韵律是否满意, 如果不满意, 则需要微调 `LLAMA`.

    建议先对 `LLAMA` 进行微调，最后再根据需要微调 `VITS Decoder`.

## LLAMA 微调
### 1. 准备数据集

```
.
├── SPK1
│   ├── 21.15-26.44.lab
│   ├── 21.15-26.44.mp3
│   ├── 27.51-29.98.lab
│   ├── 27.51-29.98.mp3
│   ├── 30.1-32.71.lab
│   └── 30.1-32.71.mp3
└── SPK2
    ├── 38.79-40.85.lab
    └── 38.79-40.85.mp3
```

你需要将数据集转为以上格式, 并放到 `data` 下, 音频后缀可以为 `.mp3`, `.wav` 或 `.flac`, 标注文件后缀建议为 `.lab`.

!!! warning
    建议先对数据集进行响度匹配, 你可以使用 [fish-audio-preprocess](https://github.com/fishaudio/audio-preprocess) 来完成这一步骤. 
    ```bash
    fap loudness-norm data-raw data --clean
    ```

### 2. 批量提取语义 token

确保你已经下载了 vqgan 权重, 如果没有, 请运行以下命令:

```bash
huggingface-cli download fishaudio/fish-speech-1 vq-gan-group-fsq-2x1024.pth --local-dir checkpoints
```

对于中国大陆用户, 可使用 mirror 下载.

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download fishaudio/fish-speech-1 vq-gan-group-fsq-2x1024.pth --local-dir checkpoints
```

随后可运行以下命令来提取语义 token:

```bash
python tools/vqgan/extract_vq.py data \
    --num-workers 1 --batch-size 16 \
    --config-name "vqgan_pretrain" \
    --checkpoint-path "checkpoints/vq-gan-group-fsq-2x1024.pth"
```

!!! note
    你可以调整 `--num-workers` 和 `--batch-size` 来提高提取速度, 但是请注意不要超过你的显存限制.  

该命令会在 `data` 目录下创建 `.npy` 文件, 如下所示:

```
.
├── SPK1
│   ├── 21.15-26.44.lab
│   ├── 21.15-26.44.mp3
│   ├── 21.15-26.44.npy
│   ├── 27.51-29.98.lab
│   ├── 27.51-29.98.mp3
│   ├── 27.51-29.98.npy
│   ├── 30.1-32.71.lab
│   ├── 30.1-32.71.mp3
│   └── 30.1-32.71.npy
└── SPK2
    ├── 38.79-40.85.lab
    ├── 38.79-40.85.mp3
    └── 38.79-40.85.npy
```

### 3. 打包数据集为 protobuf

```bash
python tools/llama/build_dataset.py \
    --input "data" \
    --output "data/quantized-dataset-ft.protos" \
    --text-extension .lab \
    --num-workers 16
```

命令执行完毕后, 你应该能在 `data` 目录下看到 `quantized-dataset-ft.protos` 文件.


### 4. 最后, 启动微调

同样的, 请确保你已经下载了 `LLAMA` 权重, 如果没有, 请运行以下命令:

```bash
huggingface-cli download fishaudio/fish-speech-1 text2semantic-sft-medium-v1.1-4k.pth --local-dir checkpoints
```

对于中国大陆用户, 可使用 mirror 下载.

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download fishaudio/fish-speech-1 text2semantic-sft-medium-v1.1-4k.pth --local-dir checkpoints
```

最后, 你可以运行以下命令来启动微调:

```bash
python fish_speech/train.py --config-name text2semantic_finetune \
    model@model.model=dual_ar_2_codebook_medium
```

!!! note
    你可以通过修改 `fish_speech/configs/text2semantic_finetune.yaml` 来修改训练参数如 `batch_size`, `gradient_accumulation_steps` 等, 来适应你的显存.

!!! note
    对于 Windows 用户, 你可以使用 `trainer.strategy.process_group_backend=gloo` 来避免 `nccl` 的问题.

训练结束后, 你可以参考 [推理](inference.md) 部分, 并携带 `--speaker SPK1` 参数来测试你的模型.

!!! info
    默认配置下, 基本只会学到说话人的发音方式, 而不包含音色, 你依然需要使用 prompt 来保证音色的稳定性.  
    如果你想要学到音色, 请将训练步数调大, 但这有可能会导致过拟合.

#### 使用 LoRA 进行微调
!!! note
    LoRA 可以减少模型过拟合的风险, 但是相应的会导致在大数据集上欠拟合.   

如果你想使用 LoRA, 请添加以下参数 `+lora@model.lora_config=r_8_alpha_16`.  

训练完成后, 你需要先将 loRA 的权重转为普通权重, 然后再进行推理.

```bash
python tools/llama/merge_lora.py \
    --llama-config dual_ar_2_codebook_medium \
    --lora-config r_8_alpha_16 \
    --llama-weight checkpoints/text2semantic-sft-medium-v1.1-4k.pth \
    --lora-weight results/text2semantic-finetune-medium-lora/checkpoints/step_000000200.ckpt \
    --output checkpoints/merged.ckpt
```

## VITS 微调
### 1. 准备数据集

```
.
├── SPK1
│   ├── 21.15-26.44.lab
│   ├── 21.15-26.44.mp3
│   ├── 27.51-29.98.lab
│   ├── 27.51-29.98.mp3
│   ├── 30.1-32.71.lab
│   └── 30.1-32.71.mp3
└── SPK2
    ├── 38.79-40.85.lab
    └── 38.79-40.85.mp3
```
!!! note
	VITS 微调目前仅支持 `.lab` 作为标签文件，不支持 `filelist` 形式.

你需要将数据集转为以上格式, 并放到 `data` 下, 音频后缀可以为 `.mp3`, `.wav` 或 `.flac`, 标注文件后缀建议为 `.lab`.

### 2. 分割训练集和验证集

```bash
python tools/vqgan/create_train_split.py data
```

该命令会在 `data` 目录下创建 `data/vq_train_filelist.txt` 和 `data/vq_val_filelist.txt` 文件, 分别用于训练和验证.  

!!! info
    对于 VITS 格式, 你可以使用 `--filelist xxx.list` 来指定文件列表.  
    请注意, `filelist` 所指向的音频文件必须也位于 `data` 文件夹下.

### 3. 启动训练

```bash
python fish_speech/train.py --config-name vits_decoder_finetune
```

!!! note
    你可以通过修改 `fish_speech/configs/vits_decoder_finetune.yaml` 来修改训练参数, 如数据集配置.

### 4. 测试音频
    
```bash
python tools/vits_decoder/inference.py \
    --checkpoint-path results/vits_decoder_finetune/checkpoints/step_000010000.ckpt \
    -i test.npy -r test.wav \
    --text "合成文本"
```

你可以查看 `fake.wav` 来判断微调效果.

## VQGAN 微调 (不推荐)

在 V1.1 版本中, 我们不再推荐使用 VQGAN 进行微调, 使用 VITS Decoder 会获得更好的表现, 但是如果你仍然想要使用 VQGAN 进行微调, 你可以参考以下步骤.

### 1. 准备数据集

```
.
├── SPK1
│   ├── 21.15-26.44.mp3
│   ├── 27.51-29.98.mp3
│   └── 30.1-32.71.mp3
└── SPK2
    └── 38.79-40.85.mp3
```

你需要将数据集转为以上格式, 并放到 `data` 下, 音频后缀可以为 `.mp3`, `.wav` 或 `.flac`.

### 2. 分割训练集和验证集

```bash
python tools/vqgan/create_train_split.py data
```

该命令会在 `data` 目录下创建 `data/vq_train_filelist.txt` 和 `data/vq_val_filelist.txt` 文件, 分别用于训练和验证.  

!!!info
    对于 VITS 格式, 你可以使用 `--filelist xxx.list` 来指定文件列表.  
    请注意, `filelist` 所指向的音频文件必须也位于 `data` 文件夹下.

### 3. 启动训练

```bash
python fish_speech/train.py --config-name vqgan_finetune
```

!!! note
    你可以通过修改 `fish_speech/configs/vqgan_finetune.yaml` 来修改训练参数, 但大部分情况下, 你不需要这么做.

### 4. 测试音频
    
```bash
python tools/vqgan/inference.py -i test.wav --checkpoint-path results/vqgan_finetune/checkpoints/step_000010000.ckpt
```

你可以查看 `fake.wav` 来判断微调效果.

!!! note
    你也可以尝试其他的 checkpoint, 我们建议你使用最早的满足你要求的 checkpoint, 他们通常在 OOD 上表现更好.
