import gradio as gr
import requests
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE


title = "Whisper JAX: The Fastest Whisper API Available ⚡️"

description = """Whisper JAX is an optimised implementation of the [Whisper model](https://huggingface.co/openai/whisper-large-v2) by OpenAI. It runs on JAX with a TPU v4-8 in the backend. Compared to PyTorch on an A100 GPU, it is over **12x** faster, making it the fastest Whisper API available.

You can submit requests to Whisper JAX through this Gradio Demo, or directly through API calls (see below). This notebook demonstrates how you can run the Whisper JAX model yourself on a TPU v2-8 in a Google Colab: TODO.
"""

API_URL = "https://whisper-jax.ngrok.io/generate/"

api_info = """## Python API call:
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
api_info = api_info.replace("{URL}", API_URL)
article = "Whisper large-v2 model by OpenAI. Backend running JAX on a TPU v4-8 through the generous support of the [TRC](https://sites.research.google/trc/about/) programme."

language_names = sorted(TO_LANGUAGE_CODE.keys())
SAMPLING_RATE = 16000


def query(payload):
    response = requests.post(API_URL, json=payload)
    return response.json(), response.status_code


def inference(inputs, language, task, return_timestamps):
    inputs = {"array": inputs[1].tolist(), "sampling_rate": inputs[0]}
    payload = {"inputs": inputs, "task": task, "return_timestamps": return_timestamps}

    if language:
        payload["language"] = language

    data, status_code = query(payload)

    if status_code == 200:
        text = data["text"]
    else:
        text = data["detail"]

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
        gr.inputs.Checkbox(default=False, label="Return timestamps"),
    ],
    outputs=[
        gr.outputs.Textbox(label="Transcription"),
        gr.outputs.Textbox(label="Timestamps"),
    ],
    examples=[["../Downloads/processed.wav", None, "transcribe", False]],
    cache_examples=False,
    allow_flagging="never",
    title=title,
    description=description,
    article=article,
).launch()
