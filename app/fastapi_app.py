import jax.numpy as jnp
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE

from whisper_jax import FlaxWhisperPipline


checkpoint = "openai/whisper-tiny"

pipeline = FlaxWhisperPipline(checkpoint, dtype=jnp.bfloat16)
pipeline.shard_params()

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


language_codes = [f"<|{lang_id}|>" for lang_id in TO_LANGUAGE_CODE.values()]


def check_inputs(inputs, language, task, return_timestamps):
    # required pre-processing to handle different input types efficiently over requests
    if isinstance(inputs, dict):
        if not ("sampling_rate" in inputs and "array" in inputs):
            raise HTTPException(
                status_code=404,
                detail=(
                    "When passing a dictionary as inputs, the dict needs to contain an "
                    '"array" key containing the numpy array representing the audio, and a "sampling_rate" key, '
                    "containing the sampling_rate associated with that array"
                ),
            )

        if isinstance(inputs["array"], str):
            inputs["array"] = np.fromstring(inputs["array"], dtype=np.int16)

        if not isinstance(inputs["array"], np.ndarray):
            raise HTTPException(status_code=404, detail=f"We expect a numpy ndarray as input, got `{type(inputs)}`")

        if len(inputs.shape) != 1:
            raise ValueError("We expect a single channel audio input for AutomaticSpeechRecognitionPipeline")

    if language is not None:
        if not isinstance("language", str) or language not in language_codes:
            raise HTTPException(
                status_code=404, detail=("language argument should be in ")
            )  # TODO(SG): handle language as string

    if task is not None:
        if not isinstance("task", str) or task not in ["transcribe", "translate"]:
            raise HTTPException(
                status_code=404, detail=(f"task argument should be either" f'"transcribe" or "translate", got {task}.')
            )

    if return_timestamps is not None:
        if not isinstance(return_timestamps, bool):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"return_timestamps should be a boolean value of either 'True' or 'False', got {return_timestamps}"
                ),
            )


@app.post("/generate/")
async def generate(request: Request):
    content = await request.json()
    inputs = content.get("inputs", None)
    language = content.get("language", None)
    task = content.get("task", "transcribe")
    return_timestamps = content.get("return_timestamps", False)

    check_inputs(inputs, language, task, return_timestamps)

    generation = pipeline(inputs, language=language, task=task, return_timestamps=return_timestamps)
    out = [generation]
    return out
