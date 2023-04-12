import base64

import jax.numpy as jnp
import numpy as np
import pytube
import requests
from fastapi import FastAPI, HTTPException, Request
from jax.experimental.compilation_cache import compilation_cache as cc
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from transformers.pipelines.audio_utils import ffmpeg_read

from whisper_jax import FlaxWhisperPipline


cc.initialize_cache("./jax_cache")
checkpoint = "openai/whisper-large-v2"
batch_size = 16
chunk_length_s = 30

pipeline = FlaxWhisperPipline(checkpoint, dtype=jnp.bfloat16)

language_codes = {lang: f"<|{TO_LANGUAGE_CODE[lang]}|>" for lang in TO_LANGUAGE_CODE}
generation_config = pipeline.model.generation_config

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


def download_youtube(yt_url, max_filesize=50.0):
    yt = pytube.YouTube(yt_url)
    stream = yt.streams.filter(only_audio=True)[0]

    if stream.filesize_mb > max_filesize:
        raise HTTPException(
            status_code=418,
            detail=f"Maximum YouTube file size is {max_filesize}MB, got {stream.filesize_mb:.2f}MB.",
        )

    stream.download(filename="audio.mp3")

    with open("audio.mp3", "rb") as f:
        inputs = f.read()

    inputs = ffmpeg_read(inputs, pipeline.feature_extractor.sampling_rate)
    inputs = {"array": inputs, "sampling_rate": pipeline.feature_extractor.sampling_rate}
    return inputs


def check_inputs(inputs, language, task, return_timestamps):
    # required pre-processing to handle different input types efficiently over requests
    if isinstance(inputs, str):
        if inputs.startswith("http://") or inputs.startswith("https://"):
            if "youtu" in inputs:
                inputs = download_youtube(inputs)
            else:
                # We need to actually check for a real protocol, otherwise it's impossible to use a local file
                # like http_huggingface_co.png
                inputs = requests.get(inputs).content
        else:
            with open(inputs, "rb") as f:
                inputs = f.read()

    if isinstance(inputs, bytes):
        inputs = ffmpeg_read(inputs, pipeline.feature_extractor.sampling_rate)

    if isinstance(inputs, dict):
        if not ("sampling_rate" in inputs and "array" in inputs):
            raise HTTPException(
                status_code=418,
                detail=(
                    "When passing a dictionary as inputs, the dictionary needs to contain an "
                    '"array" key containing the numpy array representing the audio, and a "sampling_rate" key '
                    "containing the sampling rate associated with the audio array."
                ),
            )

        if isinstance(inputs["array"], list):
            audio = np.array(inputs["array"])
            inputs["array"] = (audio - np.mean(audio)) / np.std(audio)

        if isinstance(inputs["array"], str):
            inputs["array"] = np.frombuffer(base64.b64decode(inputs["array"]), dtype=np.float32)

        if not isinstance(inputs["array"], np.ndarray):
            raise HTTPException(
                status_code=418, detail=f"We expect a numpy ndarray as input, got {type(inputs['array'])}"
            )

        if len(inputs["array"].shape) != 1:
            raise HTTPException(
                status_code=418,
                detail=f"We expect a single channel audio input for the Flax Whisper API, got {len(inputs['array'].shape)} channels.",
            )
    else:
        raise HTTPException(
            status_code=418,
            detail=f"We expect an audio input in the form of bytes or dictionary, but got {type(inputs)}.",
        )

    language_token = None
    if language is not None:
        if not isinstance(language, str):
            raise HTTPException(
                status_code=418,
                detail=f"Unsupported language: {language}. Language should be one of: {list(TO_LANGUAGE_CODE.keys())}.",
            )
        language = language.lower()
        if language in generation_config.lang_to_id.keys():
            language_token = language
        elif language in TO_LANGUAGE_CODE.values():
            language_token = f"<|{language}|>"
        elif language in TO_LANGUAGE_CODE.keys():
            language_token = f"<|{TO_LANGUAGE_CODE[language]}|>"
        else:
            if len(language) == 2:
                # ISO 639-1 language code
                acceptable_languages = list(TO_LANGUAGE_CODE.values())
            elif "<" in language or "|" in language or ">" in language:
                # generation config language code
                acceptable_languages = list(generation_config.lang_to_id.keys())
            else:
                # language passed as a string
                acceptable_languages = list(TO_LANGUAGE_CODE.keys())
            raise HTTPException(
                status_code=418,
                detail=f"Unsupported language: {language}. Language should be one of:" f" {acceptable_languages}.",
            )

    if task is not None:
        if not isinstance("task", str) or task not in ["transcribe", "translate"]:
            raise HTTPException(
                status_code=418, detail=f"Unsupported task {task}. Task should be one of: ['transcribe', 'translate']"
            )

    if return_timestamps is not None:
        if not isinstance(return_timestamps, bool):
            raise HTTPException(
                status_code=418,
                detail=(
                    f"return_timestamps should be a boolean value of either 'True' or 'False', got {return_timestamps}"
                ),
            )

    return inputs, language_token, task, return_timestamps


@app.post("/generate/")
async def generate(request: Request):
    content = await request.json()
    inputs = content.get("inputs", None)
    language = content.get("language", None)
    task = content.get("task", "transcribe")
    return_timestamps = content.get("return_timestamps", False)

    inputs, language_token, task, return_timestamps = check_inputs(inputs, language, task, return_timestamps)

    generation = pipeline(
        inputs,
        language=language,
        task=task,
        return_timestamps=return_timestamps,
        batch_size=batch_size,
        chunk_length_s=chunk_length_s,
    )
    return generation


@app.post("/generate_from_features/")
async def generate_from_features(request: Request):
    content = await request.json()
    batch = content.get("batch", None)
    feature_shape = content.get("feature_shape", None)
    language = content.get("language", None)
    task = content.get("task", "transcribe")
    return_timestamps = content.get("return_timestamps", False)

    batch["input_features"] = np.frombuffer(base64.b64decode(batch["input_features"]), dtype=np.float32).reshape(
        feature_shape
    )

    generation = pipeline.forward(
        batch, batch_size=batch_size, language=language, task=task, return_timestamps=return_timestamps
    )
    generation["tokens"] = generation["tokens"].tolist()

    return generation
