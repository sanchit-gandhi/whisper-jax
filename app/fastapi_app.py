import jax.numpy as jnp
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from transformers.pipelines.audio_utils import ffmpeg_read

from whisper_jax import FlaxWhisperPipline


# TODO(SG): switch to large checkpoint
checkpoint = "openai/whisper-tiny"

pipeline = FlaxWhisperPipline(checkpoint, dtype=jnp.bfloat16)
pipeline.shard_params()

language_codes = {lang: f"<|{TO_LANGUAGE_CODE[lang]}|>" for lang in TO_LANGUAGE_CODE}
generation_config = pipeline.model.generation_config

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


def check_inputs(inputs, language, task, return_timestamps):
    # required pre-processing to handle different input types efficiently over requests
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
            audio = np.array(inputs["array"], dtype=np.int16)
            inputs["array"] = (audio - np.mean(audio)) / np.std(audio)

        if not isinstance(inputs["array"], np.ndarray):
            raise HTTPException(
                status_code=418, detail=f"We expect a numpy ndarray as input, got {str(type(inputs['array']))}"
            )

        if len(inputs["array"].shape) != 1:
            raise HTTPException(
                status_code=418,
                detail=f"We expect a single channel audio input for the Flax Whisper API, got {str(len(inputs['array'].shape))} channels.",
            )
    """else:
        raise HTTPException(status_code=418, detail=
        f"We expect an audio input in the form of bytes or dictionary, but got {str(type(inputs))}.")"""

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

    generation = pipeline(inputs, language=language, task=task, return_timestamps=return_timestamps)
    return generation
