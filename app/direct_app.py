import math
import os
from multiprocessing import Pool

import gradio as gr
import jax.numpy as jnp
import numpy as np
import pytube
from jax.experimental.compilation_cache import compilation_cache as cc
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from transformers.pipelines.audio_utils import ffmpeg_read

from whisper_jax import FlaxWhisperPipline


cc.initialize_cache("./jax_cache")
checkpoint = "openai/whisper-large-v2"
BATCH_SIZE = 16
CHUNK_LENGTH_S = 30
NUM_PROC = 16
FILE_LIMIT_MB = 1000

title = "Whisper JAX: The Fastest Whisper API ⚡️"

description = """Whisper JAX is an optimised implementation of the [Whisper model](https://huggingface.co/openai/whisper-large-v2) by OpenAI. It runs on JAX with a TPU v4-8 in the backend. Compared to PyTorch on an A100 GPU, it is over [**70x faster**](https://github.com/sanchit-gandhi/whisper-jax#benchmarks), making it the fastest Whisper API available.

Note that at peak times, you may find yourself in the queue for this demo. When you submit a request, your queue position will be shown in the top right-hand side of the demo pane. Once you reach the front of the queue, your audio file will be transcribed, with the progress displayed on the demo pane. For details on creating your own inference endpoint, refer to the [instructions](https://github.com/sanchit-gandhi/whisper-jax#creating-an-endpoint) on the Whisper JAX repository.
"""

article = "Whisper large-v2 model by OpenAI. Backend running JAX on a TPU v4-8 through the generous support of the [TRC](https://sites.research.google/trc/about/) programme. Whisper JAX [code](https://github.com/sanchit-gandhi/whisper-jax) and Gradio demo by 🤗 Hugging Face."

language_names = sorted(TO_LANGUAGE_CODE.keys())


def identity(batch):
    return batch


if __name__ == "__main__":
    pipeline = FlaxWhisperPipline(checkpoint, dtype=jnp.bfloat16, batch_size=BATCH_SIZE)
    stride_length_s = CHUNK_LENGTH_S / 6
    chunk_len = round(CHUNK_LENGTH_S * pipeline.feature_extractor.sampling_rate)
    stride_left = stride_right = round(stride_length_s * pipeline.feature_extractor.sampling_rate)
    step = chunk_len - stride_left - stride_right
    pool = Pool(NUM_PROC)

    def tqdm_generate(inputs: dict, task: str, return_timestamps: bool, progress: gr.Progress):
        inputs_len = inputs["array"].shape[0]
        all_chunk_start_idx = np.arange(0, inputs_len, step)
        num_samples = len(all_chunk_start_idx)
        num_batches = math.ceil(num_samples / BATCH_SIZE)
        dummy_batches = list(range(num_batches))  # Gradio progress bar not compatible with generator, see https://github.com/gradio-app/gradio/issues/3841

        dataloader = pipeline.preprocess_batch(inputs, chunk_length_s=CHUNK_LENGTH_S, batch_size=BATCH_SIZE)
        progress(0, desc="Pre-processing audio file...")
        dataloader = pool.map(identity, dataloader)  # TODO(SG): wrap this in a progress bar once Gradio progress bar bug is fixed

        model_outputs = []
        # iterate over our chunked audio samples
        for batch, _ in zip(dataloader, progress.tqdm(dummy_batches, desc="Transcribing...")):
            model_outputs.append(
                pipeline.forward(batch, batch_size=BATCH_SIZE, task=task, return_timestamps=return_timestamps)
            )

        post_processed = pipeline.postprocess(model_outputs, return_timestamps=return_timestamps)
        timestamps = post_processed.get("chunks")
        return post_processed["text"], timestamps

    def transcribe_chunked_audio(inputs, task, return_timestamps, progress=gr.Progress()):
        progress(0, desc="Loading audio file...")
        file_size_mb = os.stat(inputs).st_size / (1024 * 1024)
        if file_size_mb > FILE_LIMIT_MB:
            return (
                f"ERROR: File size exceeds file size limit. Got file of size {file_size_mb:.2f}MB for a limit of {FILE_LIMIT_MB}MB.",
                None,
            )

        with open(inputs, "rb") as f:
            inputs = f.read()

        inputs = ffmpeg_read(inputs, pipeline.feature_extractor.sampling_rate)
        inputs = {"array": inputs, "sampling_rate": pipeline.feature_extractor.sampling_rate}
        text, timestamps = tqdm_generate(inputs, task=task, return_timestamps=return_timestamps, progress=progress)
        return text, timestamps

    def _return_yt_html_embed(yt_url):
        video_id = yt_url.split("?v=")[-1]
        HTML_str = (
            f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
            " </center>"
        )
        return HTML_str

    def download_youtube(yt_url, max_filesize=50.0):
        yt = pytube.YouTube(yt_url)
        stream = yt.streams.filter(only_audio=True)[0]

        if stream.filesize_mb > max_filesize:
            raise ValueError(
                f"Maximum YouTube file size is {max_filesize}MB, got {stream.filesize_mb:.2f}MB.",
            )

        stream.download(filename="audio.mp3")

        with open("audio.mp3", "rb") as f:
            inputs = f.read()

        inputs = ffmpeg_read(inputs, pipeline.feature_extractor.sampling_rate)
        inputs = {"array": inputs, "sampling_rate": pipeline.feature_extractor.sampling_rate}
        return inputs

    def transcribe_youtube(yt_url, task, return_timestamps, progress=gr.Progress()):
        progress(0, desc="Loading audio file...")
        html_embed_str = _return_yt_html_embed(yt_url)
        inputs = download_youtube(yt_url)
        text, timestamps = tqdm_generate(inputs, task=task, return_timestamps=return_timestamps, progress=progress)
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

    demo.queue(concurrency_count=3, max_size=5)
    demo.launch(show_api=False)
