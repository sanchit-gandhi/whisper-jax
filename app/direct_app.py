import math
import os
import time
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
NUM_PROC = 8
FILE_LIMIT_MB = 1000
YT_ATTEMPT_LIMIT = 3

title = "Whisper JAX: The Fastest Whisper API ⚡️"

description = """Whisper JAX is an optimised implementation of the [Whisper model](https://huggingface.co/openai/whisper-large-v2) by OpenAI. It runs on JAX with a TPU v4-8 in the backend. Compared to PyTorch on an A100 GPU, it is over [**70x faster**](https://github.com/sanchit-gandhi/whisper-jax#benchmarks), making it the fastest Whisper API available.

Note that at peak times, you may find yourself in the queue for this demo. When you submit a request, your queue position will be shown in the top right-hand side of the demo pane. Once you reach the front of the queue, your audio file will be transcribed, with the progress displayed through a progress bar. 

To skip the queue, you may wish to create your own inference endpoint, details for which can be found in the [Whisper JAX repository](https://github.com/sanchit-gandhi/whisper-jax#creating-an-endpoint).
"""

article = "Whisper large-v2 model by OpenAI. Backend running JAX on a TPU v4-8 through the generous support of the [TRC](https://sites.research.google/trc/about/) programme. Whisper JAX [code](https://github.com/sanchit-gandhi/whisper-jax) and Gradio demo by 🤗 Hugging Face."

language_names = sorted(TO_LANGUAGE_CODE.keys())


def identity(batch):
    return batch


# Copied from https://github.com/openai/whisper/blob/c09a7ae299c4c34c5839a76380ae407e7d785914/whisper/utils.py#L50
def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = "."):
    if seconds is not None:
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
        return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    else:
        # we have a malformed timestamp so just return it as is
        return seconds


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
        dummy_batches = list(
            range(num_batches)
        )  # Gradio progress bar not compatible with generator, see https://github.com/gradio-app/gradio/issues/3841

        dataloader = pipeline.preprocess_batch(inputs, chunk_length_s=CHUNK_LENGTH_S, batch_size=BATCH_SIZE)
        progress(0, desc="Pre-processing audio file...")
        dataloader = pool.map(identity, dataloader)

        model_outputs = []
        start_time = time.time()
        # iterate over our chunked audio samples
        for batch, _ in zip(dataloader, progress.tqdm(dummy_batches, desc="Transcribing...")):
            model_outputs.append(
                pipeline.forward(batch, batch_size=BATCH_SIZE, task=task, return_timestamps=return_timestamps)
            )
        runtime = time.time() - start_time

        post_processed = pipeline.postprocess(model_outputs, return_timestamps=return_timestamps)
        text = post_processed["text"]
        timestamps = post_processed.get("chunks")
        if timestamps is not None:
            timestamps = [
                f"[{format_timestamp(chunk['timestamp'][0])} -> {format_timestamp(chunk['timestamp'][1])}] {chunk['text']}"
                for chunk in timestamps
            ]
            text = "\n".join(str(feature) for feature in timestamps)
        return text, runtime

    def transcribe_chunked_audio(inputs, task, return_timestamps, progress=gr.Progress()):
        progress(0, desc="Loading audio file...")
        if inputs is None:
            raise gr.Error("No audio file submitted! Please upload an audio file before submitting your request.")
        file_size_mb = os.stat(inputs).st_size / (1024 * 1024)
        if file_size_mb > FILE_LIMIT_MB:
            raise gr.Error(
                f"File size exceeds file size limit. Got file of size {file_size_mb:.2f}MB for a limit of {FILE_LIMIT_MB}MB."
            )

        with open(inputs, "rb") as f:
            inputs = f.read()

        inputs = ffmpeg_read(inputs, pipeline.feature_extractor.sampling_rate)
        inputs = {"array": inputs, "sampling_rate": pipeline.feature_extractor.sampling_rate}
        text, runtime = tqdm_generate(inputs, task=task, return_timestamps=return_timestamps, progress=progress)
        return text, runtime

    def _return_yt_html_embed(yt_url):
        video_id = yt_url.split("?v=")[-1]
        HTML_str = (
            f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
            " </center>"
        )
        return HTML_str

    def transcribe_youtube(yt_url, task, return_timestamps, progress=gr.Progress(), max_filesize=75.0):
        progress(0, desc="Loading audio file...")
        html_embed_str = _return_yt_html_embed(yt_url)

        for attempt in range(YT_ATTEMPT_LIMIT):
            try:
                yt = pytube.YouTube(yt_url)
                stream = yt.streams.filter(only_audio=True)[0]
                break
            except KeyError:
                if attempt + 1 == YT_ATTEMPT_LIMIT:
                    raise gr.Error("An error occurred while loading the YouTube video. Please try again.")

        if stream.filesize_mb > max_filesize:
            raise gr.Error(f"Maximum YouTube file size is {max_filesize}MB, got {stream.filesize_mb:.2f}MB.")

        stream.download(filename="audio.mp3")

        with open("audio.mp3", "rb") as f:
            inputs = f.read()

        inputs = ffmpeg_read(inputs, pipeline.feature_extractor.sampling_rate)
        inputs = {"array": inputs, "sampling_rate": pipeline.feature_extractor.sampling_rate}
        text, runtime = tqdm_generate(inputs, task=task, return_timestamps=return_timestamps, progress=progress)
        return html_embed_str, text, runtime

    microphone_chunked = gr.Interface(
        fn=transcribe_chunked_audio,
        inputs=[
            gr.inputs.Audio(source="microphone", optional=True, type="filepath"),
            gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
            gr.inputs.Checkbox(default=False, label="Return timestamps"),
        ],
        outputs=[
            gr.outputs.Textbox(label="Transcription").style(show_copy_button=True),
            gr.outputs.Textbox(label="Transcription Time (s)"),
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
            gr.outputs.Textbox(label="Transcription").style(show_copy_button=True),
            gr.outputs.Textbox(label="Transcription Time (s)"),
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
            gr.outputs.Textbox(label="Transcription").style(show_copy_button=True),
            gr.outputs.Textbox(label="Transcription Time (s)"),
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
    demo.launch(server_name="0.0.0.0", show_api=False)
