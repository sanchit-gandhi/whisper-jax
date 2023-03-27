import os

import gradio as gr
import requests

from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from transformers.pipelines.audio_utils import ffmpeg_read

title = "Whisper JAX: The Fastest Whisper API Available ⚡️"

description = """Gradio Demo for Whisper JAX. Whisper JAX is an optimised JAX implementation of the Whisper model by OpenAI. It runs on a TPU v4-8 in the backend. Compared to PyTorch on an A100 GPU, it is over **12x** faster, making it the fastest Whisper API available.

You can submit requests to Whisper JAX through this Gradio Demo, or directly through API calls (see below).
"""

API_URL = "https://whisper-jax.ngrok.io/generate"

article = """## Python API call:
```python
import requests

response = requests.post("{URL}", json={
  "inputs": "/path/to/file/audio.mp3",
  "task": "transcribe",
  "return_timestamps": False,
}).json()

data = response["data"]
```

## Javascript API call:
```javascript
fetch("{URL}", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    data: [
      "/path/to/file/audio.mp3",
      "afrikaans",
      "transcribe",
      false,
	]
  })})
.then(r => r.json())
.then(
  r => {
    let data = r.data;
  }
)
```

## CURL API call:
```
curl -X POST -d '{"inputs": "/path/to/file/audio.mp3", "task": "transcribe", "return_timestamps": false}' {URL} -H "content-type: application/json"
```
"""
article = article.replace("{URL}", API_URL)

language_names = sorted(list(TO_LANGUAGE_CODE.keys()))


def query(files_payload, json_payload):
    response = requests.post(API_URL, data=files_payload, json=json_payload)
    return response.json()


def inference(input, language, task, return_timestamps):
    json_payload = {
        "task": task,
        "return_timestamps": return_timestamps
    }

    if language:
        json_payload["language"] = f"<|{TO_LANGUAGE_CODE[language]}|>"

    data = query(
        {"inputs": {"array": input[1], "sampling_rate": input[0]}},
        json_payload
    )

    text = data[0]["text"]

    if return_timestamps:
        timestamps = data[0]["chunks"]
    else:
        timestamps = None

    return text, timestamps


gr.Interface(
    fn=inference,
    inputs=[
        gr.inputs.Audio(source="upload", label="Input"),
        gr.inputs.Dropdown(language_names, label="Language", default=None),
        gr.inputs.Dropdown(["transcribe", "translate"], label="Task", default="transcribe"),
        gr.inputs.Checkbox(default=False, label="Return timestamps")
    ],
    outputs=[
        gr.outputs.Textbox(label="Transcription"),
        gr.outputs.Textbox(label="Timestamps"),
    ],
    examples=[["../Downloads/processed.wav", None, "transcribe", False]],
    cache_examples=False,
    title=title,
    description=description,
    article=article,
).launch()
