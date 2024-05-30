import gc
import html
import io
import os
import queue
import wave
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import gradio as gr
import numpy as np
import pyrootutils
import torch
from loguru import logger
from transformers import AutoTokenizer

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.i18n import i18n
from tools.api import decode_vq_tokens, encode_reference
from tools.llama.generate import (
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
    launch_thread_safe_queue,
)
from tools.vqgan.inference import load_model as load_decoder_model

# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"


HEADER_MD = f"""# Fish Speech

{i18n("A text-to-speech model based on VQ-GAN and Llama developed by [Fish Audio](https://fish.audio).")}  

{i18n("You can find the source code [here](https://github.com/fishaudio/fish-speech) and models [here](https://huggingface.co/fishaudio/fish-speech-1).")}  

{i18n("Related code are released under BSD-3-Clause License, and weights are released under CC BY-NC-SA 4.0 License.")}  

{i18n("We are not responsible for any misuse of the model, please consider your local laws and regulations before using it.")}  
"""

TEXTBOX_PLACEHOLDER = i18n("Put your text here.")
SPACE_IMPORTED = False


def build_html_error_message(error):
    return f"""
    <div style="color: red; 
    font-weight: bold;">
        {html.escape(str(error))}
    </div>
    """


@torch.inference_mode()
def inference(
    text,
    enable_reference_audio,
    reference_audio,
    reference_text,
    max_new_tokens,
    chunk_length,
    top_p,
    repetition_penalty,
    temperature,
    speaker,
    streaming=False,
):
    if args.max_gradio_length > 0 and len(text) > args.max_gradio_length:
        return (
            None,
            None,
            i18n("Text is too long, please keep it under {} characters.").format(
                args.max_gradio_length
            ),
        )

    # Parse reference audio aka prompt
    prompt_tokens, reference_embedding = encode_reference(
        decoder_model=decoder_model,
        reference_audio=reference_audio,
        enable_reference_audio=enable_reference_audio,
    )

    # LLAMA Inference
    request = dict(
        tokenizer=llama_tokenizer,
        device=decoder_model.device,
        max_new_tokens=max_new_tokens,
        text=text,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        compile=args.compile,
        iterative_prompt=chunk_length > 0,
        chunk_length=chunk_length,
        max_length=args.max_length,
        speaker=speaker if speaker else None,
        prompt_tokens=prompt_tokens if enable_reference_audio else None,
        prompt_text=reference_text if enable_reference_audio else None,
    )

    response_queue = queue.Queue()
    llama_queue.put(
        GenerateRequest(
            request=request,
            response_queue=response_queue,
        )
    )

    if streaming:
        yield wav_chunk_header(), None, None

    segments = []

    while True:
        result: WrappedGenerateResponse = response_queue.get()
        if result.status == "error":
            yield None, None, build_html_error_message(result.response)
            break

        result: GenerateResponse = result.response
        if result.action == "next":
            break

        text_tokens = llama_tokenizer.encode(result.text, return_tensors="pt").to(
            decoder_model.device
        )

        with torch.autocast(
            device_type=decoder_model.device.type, dtype=args.precision
        ):
            fake_audios = decode_vq_tokens(
                decoder_model=decoder_model,
                codes=result.codes,
                text_tokens=text_tokens,
                reference_embedding=reference_embedding,
            )

        fake_audios = fake_audios.float().cpu().numpy()
        segments.append(fake_audios)

        if streaming:
            yield (fake_audios * 32768).astype(np.int16).tobytes(), None, None

    if len(segments) == 0:
        return (
            None,
            None,
            build_html_error_message(
                i18n("No audio generated, please check the input text.")
            ),
        )

    # No matter streaming or not, we need to return the final audio
    audio = np.concatenate(segments, axis=0)
    yield None, (decoder_model.sampling_rate, audio), None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


inference_stream = partial(inference, streaming=True)

n_audios = 4

global_audio_list = []
global_error_list = []


def inference_wrapper(
    text,
    enable_reference_audio,
    reference_audio,
    reference_text,
    max_new_tokens,
    chunk_length,
    top_p,
    repetition_penalty,
    temperature,
    speaker,
    batch_infer_num,
):
    audios = []
    errors = []

    for _ in range(batch_infer_num):
        items = inference(
            text,
            enable_reference_audio,
            reference_audio,
            reference_text,
            max_new_tokens,
            chunk_length,
            top_p,
            repetition_penalty,
            temperature,
            speaker,
        )

        try:
            item = next(items)
        except StopIteration:
            print("No more audio data available.")

        audios.append(
            gr.Audio(value=item[1] if (item and item[1]) else None, visible=True),
        )
        errors.append(
            gr.HTML(value=item[2] if (item and item[2]) else None, visible=True),
        )

    for _ in range(batch_infer_num, n_audios):
        audios.append(
            gr.Audio(value=None, visible=False),
        )
        errors.append(
            gr.HTML(value=None, visible=False),
        )

    return None, *audios, *errors


def wav_chunk_header(sample_rate=44100, bit_depth=16, channels=1):
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)

    wav_header_bytes = buffer.getvalue()
    buffer.close()
    return wav_header_bytes


def build_app():
    with gr.Blocks(theme=gr.themes.Base()) as app:
        gr.Markdown(HEADER_MD)

        # Use light theme by default
        app.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', 'light');window.location.search = params.toString();}}",
        )

        # Inference
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label=i18n("Input Text"), placeholder=TEXTBOX_PLACEHOLDER, lines=15
                )

                with gr.Row():
                    with gr.Tab(label=i18n("Advanced Config")):
                        chunk_length = gr.Slider(
                            label=i18n("Iterative Prompt Length, 0 means off"),
                            minimum=0,
                            maximum=500,
                            value=150,
                            step=8,
                        )

                        max_new_tokens = gr.Slider(
                            label=i18n("Maximum tokens per batch, 0 means no limit"),
                            minimum=0,
                            maximum=args.max_length,
                            value=0,  # 0 means no limit
                            step=8,
                        )

                        top_p = gr.Slider(
                            label="Top-P", minimum=0, maximum=1, value=0.7, step=0.01
                        )

                        repetition_penalty = gr.Slider(
                            label=i18n("Repetition Penalty"),
                            minimum=0,
                            maximum=2,
                            value=1.5,
                            step=0.01,
                        )

                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0,
                            maximum=2,
                            value=0.7,
                            step=0.01,
                        )

                        speaker = gr.Textbox(
                            label=i18n("Speaker"),
                            placeholder=i18n("Type name of the speaker"),
                            lines=1,
                        )

                    with gr.Tab(label=i18n("Reference Audio")):
                        gr.Markdown(
                            i18n(
                                "5 to 10 seconds of reference audio, useful for specifying speaker."
                            )
                        )

                        enable_reference_audio = gr.Checkbox(
                            label=i18n("Enable Reference Audio"),
                        )
                        reference_audio = gr.Audio(
                            label=i18n("Reference Audio"),
                            type="filepath",
                        )
                        reference_text = gr.Textbox(
                            label=i18n("Reference Text"),
                            placeholder=i18n("Reference Text"),
                            lines=1,
                            value="在一无所知中，梦里的一天结束了，一个新的「轮回」便会开始。",
                        )
                    with gr.Tab(label=i18n("Batch Inference")):
                        batch_infer_num = gr.Slider(
                            label="Batch infer nums",
                            minimum=1,
                            maximum=n_audios,
                            step=1,
                            value=1,
                        )

            with gr.Column(scale=3):
                for _ in range(n_audios):
                    with gr.Row():
                        error = gr.HTML(
                            label=i18n("Error Message"),
                            visible=True if _ == 0 else False,
                        )
                        global_error_list.append(error)
                    with gr.Row():
                        audio = gr.Audio(
                            label=i18n("Generated Audio"),
                            type="numpy",
                            interactive=False,
                            visible=True if _ == 0 else False,
                        )
                        global_audio_list.append(audio)

                with gr.Row():
                    stream_audio = gr.Audio(
                        label=i18n("Streaming Audio"),
                        streaming=True,
                        autoplay=True,
                        interactive=False,
                    )
                with gr.Row():
                    with gr.Column(scale=3):
                        generate = gr.Button(
                            value="\U0001F3A7 " + i18n("Generate"), variant="primary"
                        )
                        generate_stream = gr.Button(
                            value="\U0001F3A7 " + i18n("Streaming Generate"),
                            variant="primary",
                        )
        # # Submit
        generate.click(
            inference_wrapper,
            [
                text,
                enable_reference_audio,
                reference_audio,
                reference_text,
                max_new_tokens,
                chunk_length,
                top_p,
                repetition_penalty,
                temperature,
                speaker,
                batch_infer_num,
            ],
            [stream_audio, *global_audio_list, *global_error_list],
            concurrency_limit=1,
        )

        generate_stream.click(
            inference_stream,
            [
                text,
                enable_reference_audio,
                reference_audio,
                reference_text,
                max_new_tokens,
                chunk_length,
                top_p,
                repetition_penalty,
                temperature,
                speaker,
            ],
            [stream_audio, global_audio_list[0], global_error_list[0]],
            concurrency_limit=10,
        )
    return app


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default="checkpoints/text2semantic-sft-medium-v1-4k.pth",
    )
    parser.add_argument(
        "--llama-config-name", type=str, default="dual_ar_2_codebook_medium"
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=Path,
        default="checkpoints/vq-gan-group-fsq-2x1024.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="vqgan_pretrain")
    parser.add_argument("--tokenizer", type=str, default="fishaudio/fish-speech-1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-gradio-length", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.precision = torch.half if args.half else torch.bfloat16

    logger.info("Loading Llama model...")
    llama_queue = launch_thread_safe_queue(
        config_name=args.llama_config_name,
        checkpoint_path=args.llama_checkpoint_path,
        device=args.device,
        precision=args.precision,
        max_length=args.max_length,
        compile=args.compile,
    )
    llama_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    logger.info("Llama model loaded, loading VQ-GAN model...")

    decoder_model = load_decoder_model(
        config_name=args.decoder_config_name,
        checkpoint_path=args.decoder_checkpoint_path,
        device=args.device,
    )

    logger.info("Decoder model loaded, warming up...")

    # Dry run to check if the model is loaded correctly and avoid the first-time latency
    list(
        inference(
            text="Hello, world!",
            enable_reference_audio=False,
            reference_audio=None,
            reference_text="",
            max_new_tokens=0,
            chunk_length=150,
            top_p=0.7,
            repetition_penalty=1.5,
            temperature=0.7,
            speaker=None,
        )
    )

    logger.info("Warming up done, launching the web UI...")

    app = build_app()
    app.launch(show_api=True)
