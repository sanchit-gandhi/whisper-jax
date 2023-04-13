import base64
import os
from functools import partial
from multiprocessing import Pool

import gradio as gr
import numpy as np
import requests
from processing_whisper import WhisperPrePostProcessor
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from transformers.pipelines.audio_utils import ffmpeg_read


title = "Whisper JAX: The Fastest Whisper API ⚡️"

description = """Whisper JAX is an optimised implementation of the [Whisper model](https://huggingface.co/openai/whisper-large-v2) by OpenAI. It runs on JAX with a TPU v4-8 in the backend. Compared to PyTorch on an A100 GPU, it is over [**70x faster**](https://github.com/sanchit-gandhi/whisper-jax#benchmarks), making it the fastest Whisper API available.

Note that using microphone or audio file requires the audio input to be transferred from the Gradio demo to the TPU, which for large audio files can be slow. We recommend using YouTube where possible, since this directly downloads the audio file to the TPU, skipping the file transfer step.
"""

API_URL = os.getenv("API_URL")
API_URL_FROM_FEATURES = os.getenv("API_URL_FROM_FEATURES")

article = "Whisper large-v2 model by OpenAI. Backend running JAX on a TPU v4-8 through the generous support of the [TRC](https://sites.research.google/trc/about/) programme. Whisper JAX [code](https://github.com/sanchit-gandhi/whisper-jax) and Gradio demo by 🤗 Hugging Face."

language_names = sorted(TO_LANGUAGE_CODE.keys())
CHUNK_LENGTH_S = 30
BATCH_SIZE = 16
NUM_PROC = 16
FILE_LIMIT_MB = 1000


def query(payload):
    response = requests.post(API_URL, json=payload)
    return response.json(), response.status_code


def inference(inputs, task=None, return_timestamps=False):
    payload = {"inputs": inputs, "task": task, "return_timestamps": return_timestamps}

    data, status_code = query(payload)

    if status_code == 200:
        text = data["text"]
    else:
        text = data["detail"]

    timestamps = data.get("chunks", None)

    return text, timestamps


def chunked_query(payload):
    response = requests.post(API_URL_FROM_FEATURES, json=payload)
    return response.json()


def forward(batch, task=None, return_timestamps=False):
    feature_shape = batch["input_features"].shape
    batch["input_features"] = base64.b64encode(batch["input_features"].tobytes()).decode()
    outputs = chunked_query(
        {"batch": batch, "task": task, "return_timestamps": return_timestamps, "feature_shape": feature_shape}
    )
    outputs["tokens"] = np.asarray(outputs["tokens"])
    return outputs


if __name__ == "__main__":
    processor = WhisperPrePostProcessor.from_pretrained("openai/whisper-large-v2")
    pool = Pool(NUM_PROC)

    def transcribe_chunked_audio(inputs, task, return_timestamps):
        file_size_mb = os.stat(inputs).st_size / (1024 * 1024)
        if file_size_mb > FILE_LIMIT_MB:
            return f"ERROR: File size exceeds file size limit. Got file of size {file_size_mb:.2f}MB for a limit of {FILE_LIMIT_MB}MB.", None

        with open(inputs, "rb") as f:
            inputs = f.read()

        inputs = ffmpeg_read(inputs, processor.feature_extractor.sampling_rate)
        inputs = {"array": inputs, "sampling_rate": processor.feature_extractor.sampling_rate}

        dataloader = processor.preprocess_batch(inputs, chunk_length_s=CHUNK_LENGTH_S, batch_size=BATCH_SIZE)

        try:
            model_outputs = pool.map(partial(forward, task=task, return_timestamps=return_timestamps), dataloader)
        except ValueError as err:
            # pre-processor does all the necessary compatibility checks for our audio inputs
            return err, None

        post_processed = processor.postprocess(model_outputs, return_timestamps=return_timestamps)
        timestamps = post_processed.get("chunks")
        return post_processed["text"], timestamps

    def _return_yt_html_embed(yt_url):
        video_id = yt_url.split("?v=")[-1]
        HTML_str = (
            f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
            " </center>"
        )
        return HTML_str

    def transcribe_youtube(yt_url, task, return_timestamps):
        html_embed_str = _return_yt_html_embed(yt_url)

        text, timestamps = inference(inputs=yt_url, task=task, return_timestamps=return_timestamps)

        return html_embed_str, text, timestamps

    microphone_chunked = gr.Interface(
        fn=transcribe_chunked_audio,
        inputs=[
            gr.inputs.Audio(source="microphone", optional=True, type="filepath"),
            gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
            gr.inputs.Checkbox(default=False, label="Return timestamps"),
        ],
        outputs=[
            gr.outputs.Textbox(label="Transcription"),
            gr.outputs.Textbox(label="Timestamps"),
        ],
        allow_flagging="never",
        title=title,
        description=description,
        article=article,
    )

    audio_chunked = gr.Interface(
        fn=transcribe_chunked_audio,
        inputs=[
            gr.inputs.Audio(source="upload", optional=True, label="Audio file", type="filepath"),
            gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
            gr.inputs.Checkbox(default=False, label="Return timestamps"),
        ],
        outputs=[
            gr.outputs.Textbox(label="Transcription"),
            gr.outputs.Textbox(label="Timestamps"),
        ],
        allow_flagging="never",
        title=title,
        description=description,
        article=article,
    )

    youtube = gr.Interface(
        fn=transcribe_youtube,
        inputs=[
            gr.inputs.Textbox(lines=1, placeholder="Paste the URL to a YouTube video here", label="YouTube URL"),
            gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
            gr.inputs.Checkbox(default=False, label="Return timestamps"),
        ],
        outputs=[
            gr.outputs.HTML(label="Video"),
            gr.outputs.Textbox(label="Transcription"),
            gr.outputs.Textbox(label="Timestamps"),
        ],
        allow_flagging="never",
        title=title,
        examples=[["https://www.youtube.com/watch?v=m8u-18Q0s7I", "transcribe", False]],
        cache_examples=False,
        description=description,
        article=article,
    )

    demo = gr.Blocks()

    with demo:
        gr.TabbedInterface([microphone_chunked, audio_chunked, youtube], ["Microphone", "Audio File", "YouTube"])

    demo.queue(max_size=3)
    demo.launch(show_api=False)
