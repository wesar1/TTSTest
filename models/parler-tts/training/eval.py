import torch
import evaluate
from transformers import AutoModel, AutoProcessor, pipeline, WhisperForConditionalGeneration, WhisperTokenizer, WhisperTokenizerFast


def clap_similarity(clap_model_name_or_path, texts, audios, device):
    clap = AutoModel.from_pretrained(clap_model_name_or_path)
    clap_processor = AutoProcessor.from_pretrained(clap_model_name_or_path)
    clap_inputs = clap_processor(text=texts, audios=audios, padding=True, return_tensors="pt").to(device)
    clap.to(device)
    with torch.no_grad():
        text_features = clap.get_text_features(
            clap_inputs["input_ids"], attention_mask=clap_inputs.get("attention_mask", None)
        )
        audio_features = clap.get_audio_features(clap_inputs["input_features"])

        cosine_sim = torch.nn.functional.cosine_similarity(audio_features, text_features, dim=1, eps=1e-8)

    clap.to("cpu")
    clap_inputs.to("cpu")
    return cosine_sim.mean().to("cpu")


def wer(asr_model_name_or_path, prompts, audios, device, per_device_eval_batch_size, sampling_rate):
    metric = evaluate.load("wer")
    asr_pipeline = pipeline(model=asr_model_name_or_path, device=device)

    return_language = None
    if isinstance(asr_pipeline.model, WhisperForConditionalGeneration):
        return_language = True

    transcriptions = asr_pipeline(
        [{"raw": audio, "sampling_rate": sampling_rate} for audio in audios],
        batch_size=int(per_device_eval_batch_size),
        return_language=return_language,
    )

    if isinstance(asr_pipeline.tokenizer, (WhisperTokenizer, WhisperTokenizerFast)):
        tokenizer = asr_pipeline.tokenizer
    else:
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")

    english_normalizer = tokenizer.normalize
    basic_normalizer = tokenizer.basic_normalize

    normalized_predictions = []
    normalized_references = []

    for pred, ref in zip(transcriptions, prompts):
        normalizer = english_normalizer if hasattr(pred, "language") and pred["language"] == "english" else basic_normalizer
        norm_ref = normalizer(ref)
        if len(norm_ref) > 0:
            norm_pred = normalizer(pred["text"])
            normalized_predictions.append(norm_pred)
            normalized_references.append(norm_ref)

    word_error = 100 * metric.compute(predictions=normalized_predictions, references=normalized_references)

    return word_error, [t["text"] for t in transcriptions]
