# Whisper JAX

This repository contains optimised JAX code for OpenAI's [Whisper Model](https://arxiv.org/abs/2212.04356), largely built 
on the 🤗 Hugging Face Transformers Whisper implementation. Compared to OpenAI's PyTorch code, Whisper JAX runs over **70x** 
faster, making it the fastest Whisper implementation available.

The JAX code is compatible on CPU, GPU and TPU. The following Google Colab demonstrates how Whisper JAX can be run in a 
Google Colab using a TPU v2-8:

## Installation

Whisper JAX was tested using Python 3.9 and JAX version 0.4.5. Installation assumes that you already have the latest 
version of the JAX package installed on your device (#TODO include JAX instructions or not?)


### JAX
First install JAX according to your device type (CPU/GPU/TPU) from the official JAX installation guide: https://github.com/google/jax#installation

#### CPU
```
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
```

#### GPU
JAX with NVIDIA GPU support can be installed with CUDA and CUDNN using pip wheels:
```
pip install --upgrade pip
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

#### Cloud TPU
JAX provides pre-built wheels for Google Cloud TPU:
```
pip install --upgrade pip
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

#### Colab TPU
JAX comes pre-installed on a Colab TPU. Simply run the following code in a Python window to initialise JAX on the TPU:

```python
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()
```

If you encounter difficulties installing the correct version of JAX, refer to the official JAX installation guide: https://github.com/google/jax#installation.

### Whisper JAX
Once the appropriate version of JAX has been installed, Whisper JAX can be installed through pip:
```
pip install git+https://github.com/sanchit-gandhi/whisper-jax
```

To update the Whisper JAX package to the latest version, simply run:
```
pip install --upgrade --no-deps --force-reinstall git+https://github.com/sanchit-gandhi/whisper-jax
```

## Pipeline Usage

The recommended way of running Whisper JAX is through the [`FlaxWhisperPipline`](https://github.com/sanchit-gandhi/whisper-jax/blob/main/whisper_jax/pipeline.py#L57) abstraction class. This class handles all
the necessary pre- and post-processing, as well as wrapping the generate method for data parallelism across accelerator devices.

Whisper JAX makes use of JAX's `pmap` function for data parallelism across GPU/TPU devices. This function is _Just In Time (JIT)_ 
compiled the first time it is called. Thereafter, the function will be _cached_, enabling it to be run in super-fast time:

```python
from whisper_jax import FlaxWhisperPipline

# instantiate pipeline
pipeline = FlaxWhisperPipline("openai/whisper-tiny")

# JIT compile the forward call - slow, but we only do once
text = pipeline("audio.mp3")

# used cached function thereafter - super fast!!
text = pipeline("audio.mp3")
```

### Half-Precision

The model computation can be run in half-precision by passing the dtype argument when instantiating the pipeline. This will 
speed-up the computation quite considerably by storing intermediate tensors in half-precision. There is no change to the precision 
of the model weights.

For most GPUs, the dtype should be set to `jnp.float16`. For A100 GPUs or TPUs, the dtype should be set to `jnp.bfloat16`:
```python
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

# instantiate pipeline in bfloat16
pipeline = FlaxWhisperPipline("openai/whisper-tiny", dtype=jnp.bfloat16)
```

### Batching
Whisper JAX also provides the option of _batching_ a single audio input across accelerator devices. The audio is first 
chunked into 30 second segments, and then chunks dispatched to the model to be transcribed in parallel. The resulting 
transcriptions are stitched back together at the boundaries to give a single, uniform transcription. In practice, batching 
provides a 10x speed-up compared to transcribing the audio samples sequentially, with a less than 1% penalty to the WER[^1], provided the batch size is selected large enough. 

To enable batching, pass the `batch_size` parameter when you instantiate the pipeline:

```python
from whisper_jax import FlaxWhisperPipline

# instantiate pipeline with batching
pipeline = FlaxWhisperPipline("openai/whisper-tiny", batch_size=8)
```

### Task

By default, the pipeline transcribes the audio file in the language it was spoken in. For speech translation, set the 
`task` argument to `"translate"`:

```python
# translate
text = pipeline("audio.mp3", task="translate")
```

### Timestamps

The [`FlaxWhisperPipline`](https://github.com/sanchit-gandhi/whisper-jax/blob/main/whisper_jax/pipeline.py#L57) also supports timestamp prediction. Note that enabling timestamps will require a second JIT compilation of the 
forward call, this time including the timestamp outputs:

```python
# transcribe and return timestamps
outputs = pipeline("audio.mp3",  task="transcribe", return_timestamps=True)
text = outputs["text"]  # transcription
chunks = outputs["chunks"]  # transcription + timestamps
```

### Putting it all together
In the following code snippet, we instantiate the model in bfloat16 precision with batching enabled, and transcribe the audio file 
returning timestamps tokens: 

```python
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

# instantiate pipeline with bfloat16 and enable batching
pipeline = FlaxWhisperPipline("openai/whisper-tiny", dtype=jnp.bfloat16, batch_size=8)

# transcribe and return timestamps
outputs = pipeline("audio.mp3",  task="transcribe", return_timestamps=True)
```

## Model Usage

The Whisper JAX model can use on a more granular level in much the same way as the original Hugging Face 
Transformers implementation. This requires the Whisper processor to be loaded separately to the model to handle the
pre- and post-processing, and the generate function to be wrapped using `pmap` by hand:

```python
import jax.numpy as jnp
from datasets import load_dataset
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from jax import device_get, pmap
from transformers import WhisperProcessor

from whisper_jax import FlaxWhisperForConditionalGeneration

# load the processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-tiny", dtype=jnp.bfloat16, _do_init=False,
)

def generate_fn(input_features):
    pred_ids = model.generate(
        input_features, task="transcribe", return_timestamps=False, max_length=model.config.max_length, params=params,
    )
    return pred_ids.sequences

# pmap the generate function for data parallelism
p_generate = pmap(generate_fn, "input_features")
# replicate the parameters across devices
params = replicate(params)

# load a dummy sample from the LibriSpeech dataset
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]

# pre-process: convert the audio array to log-mel input features
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="np").input_features
# replicate the input features across devices for DP
input_features = shard(input_features)

# run the forward pass (JIT compiled the first time it is called)
pred_ids = p_generate(input_features)
output_ids = device_get(pred_ids.reshape(-1, model.config.max_length))

# post-process: convert tokens ids to text string
transcription = processor.batch_decode(pred_ids, skip_special_tokens=True)
```

## Available Models and Languages
All Whisper models on the HuggingFace Hub with Flax weights are compatible with Whisper JAX. This includes, but is not limited to,
the official OpenAI Whisper checkpoints:

| Size     | Parameters | English-only                                         | Multilingual                                        |
|----------|------------|------------------------------------------------------|-----------------------------------------------------|
| tiny     | 39 M       | [✓](https://huggingface.co/openai/whisper-tiny.en)   | [✓](https://huggingface.co/openai/whisper-tiny)     |
| base     | 74 M       | [✓](https://huggingface.co/openai/whisper-base.en)   | [✓](https://huggingface.co/openai/whisper-base)     |
| small    | 244 M      | [✓](https://huggingface.co/openai/whisper-small.en)  | [✓](https://huggingface.co/openai/whisper-small)    |
| medium   | 769 M      | [✓](https://huggingface.co/openai/whisper-medium.en) | [✓](https://huggingface.co/openai/whisper-medium)   |
| large    | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large)    |
| large-v2 | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large-v2) |

Should you wish to use a fine-tuned Whisper checkpoint in Whisper JAX, you should first convert the PyTorch weights to Flax.
This is straightforward through use of the `from_pt` argument, which will convert the PyTorch state dict to a frozen Flax 
parameter dictionary on the fly. You can then push the converted Flax weights to the Hub to be used directly in Flax 
the next time they are required. Note that converting weights from PyTorch to Flax requires both PyTorch and Flax to be installed.

For example, to convert the fine-tuned checkpoint `"sanchit-gandhi/whisper-small-hi"` from the blog post [Fine-Tuning Whisper](https://huggingface.co/blog/fine-tune-whisper):
```python
from whisper_jax import FlaxWhisperForConditionalGeneration

checkpoint_id = "sanchit-gandhi/whisper-small-hi"
# convert PyTorch weights to Flax
model = FlaxWhisperForConditionalGeneration.from_pretrained(checkpoint_id, from_pt=True)
# push converted weights to the Hub
model.push_to_hub(checkpoint_id)

# now we can load the Flax weights directly as required
model, params = FlaxWhisperForConditionalGeneration.from_pretrained(checkpoint_id, _do_init=False)
```

## Advanced Usage
More advanced users may wish to explore different parallelisation techniques. The Whisper JAX code is
built on-top of the [T5x codebase](https://github.com/google-research/t5x), meaning it can be run using model, activation, and data parallelism using the T5x 
partitioning convention. To use T5x partitioning, the logical axis rules and number of model partitions must be defined.
For more details, the user is referred to the official T5x partitioning guide: https://github.com/google-research/t5x/blob/main/docs/usage/partitioning.md

### Pipeline
The following code snippet demonstrates how data parallelism can be achieved using the pipeline `shard_params` method in 
an entirely equivalent way to `pmap`:

```python
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

# 2D parameter and activation partitioning for DP
logical_axis_rules_dp = (
    ("batch", "data"),
    ("mlp", None),
    ("heads", None),
    ("vocab", None),
    ("embed", None),
    ("embed", None),
    ("joined_kv", None),
    ("kv", None),
    ("length", None),
    ("num_mel", None),
    ("channels", None),
)

pipeline = FlaxWhisperPipline("openai/whisper-tiny", dtype=jnp.bfloat16, batch_size=8)
pipeline.shard_params(num_mp_partitions=1, logical_axis_rules=logical_axis_rules_dp)
```

### Model
It is also possible to use the Whisper JAX model with T5x partitioning by defining a T5x inference state and T5x partitioner:

```python
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze
from jax.sharding import PartitionSpec as P

from whisper_jax import FlaxWhisperForConditionalGeneration, InferenceState, PjitPartitioner


# 2D parameter and activation partitioning for DP
logical_axis_rules_dp = [
    ("batch", "data"),
    ("mlp", None),
    ("heads", None),
    ("vocab", None),
    ("embed", None),
    ("embed", None),
    ("joined_kv", None),
    ("kv", None),
    ("length", None),
    ("num_mel", None),
    ("channels", None),
]

model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-tiny",
    _do_init=False,
    dtype=jnp.bfloat16,
)


def init_fn():
    input_shape = (1, 80, 3000)

    input_features = jnp.zeros(input_shape, dtype="f4")
    input_features = input_features.at[(..., -1)].set(model.config.eos_token_id)

    decoder_input_ids = jnp.zeros((input_shape[0], 1), dtype="i4")
    decoder_attention_mask = jnp.ones_like(decoder_input_ids)

    batch_size, sequence_length = decoder_input_ids.shape
    decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

    rng = jax.random.PRNGKey(0)
    init_params = model.module.init(
        rng,
        input_features=input_features,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        decoder_position_ids=decoder_position_ids,
        return_dict=False,
    )
    return init_params


# Axis names metadata
param_axes = jax.eval_shape(init_fn)["params_axes"]

# Create InferenceState, since the partitioner expects it
state = InferenceState(
    step=jnp.array(0),
    params=freeze(model.params_shape_tree),
    params_axes=freeze(param_axes),
    flax_mutables=None,
    flax_mutables_axes=param_axes,
)

# Define the pjit partitioner with 1 model partition
partitioner = PjitPartitioner(
    num_partitions=1,
    logical_axis_rules=logical_axis_rules_dp,
)

mesh_axes = partitioner.get_mesh_axes(state)
params_spec = mesh_axes.params

p_shard_params = partitioner.partition(model.to_bf16, (params_spec,), params_spec)


def generate(params, input_features):
    output_ids = model.generate(input_features, params=params, max_length=model.config.max_length).sequences
    return output_ids


p_generate = partitioner.partition(
    generate,
    in_axis_resources=(params_spec, P("data")),
    out_axis_resources=P("data"),
)

# This will auto-magically run in mesh context
params = p_shard_params(freeze(params))
```

## Benchmarks

We compare Whisper JAX to the official [OpenAI implementation](https://github.com/openai/whisper) and the [🤗 Transformers 
implementation](https://huggingface.co/docs/transformers/model_doc/whisper). We benchmark the models on audio samples of 
increasing length and report the average inference time over 10 repeat runs. OpenAI and Transformers both run in PyTorch 
on GPU. Whisper JAX runs in JAX on TPU. OpenAI transcribes the audio sequentially in the order it is spoken. Both Transformers
and Whisper JAX use a batching algorithm, where chunks of audio are batched together and transcribed in parallel 
(see section [Batching](#batching)).

|           | OpenAI  | Transformers | Whisper JAX |
|-----------|---------|--------------|-------------|
|           |         |              |             |
| Framework | PyTorch | PyTorch      | JAX         |
| Backend   | GPU     | GPU          | TPU         |
|           |         |              |             |
| 1 min     | 13.8    | 4.54         | 0.45        |
| 10 min    | 108.3   | 20.2         | 2.01        |
| 1 hour    | 1001.0  | 126.1        | 13.8        |
|           |         |              |             |

## Creating an Endpoint

The Whisper JAX model is running as a demo on the Hugging Face Hub:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/sanchit-gandhi/whisper-jax)

However, at peak times there may be a queue of users that limit how quickly your audio input is transcribed. In this case,
you may benefit from creating your own inference endpoint, such that you have unrestricted access to the Whisper JAX model.

We provide all the necessary code for creating an inference endpoint. To obtain this code, first clone the repository on the GPU/TPU on which you want to host the endpoint:
```
git clone https://github.com/sanchit-gandhi/whisper-jax
```

And then install Whisper JAX from source, with the required additional endpoint dependencies:
```
cd whisper-jax
pip install -e .["endpoint"]
```

The Python script [`fastapi_app.py`](app/fastapi_app.py) contains the code to launch a FastAPI app with the Whisper large-v2 model.
By default, it uses a batch size of 16 and bfloat16 half-precision. You should update these parameters depending on your 
GPU/TPU device (as explained in the sections on [Half-precision](#half-precision) and [Batching](#batching)).

You can launch the FastAPI app through Uvicorn using the bash script [`launch_app.sh`](app/launch_app.sh):
```
bash launch_app.sh
```

This will open the port 8000 for the FastAPI app. To direct network requests to the FastAPI app, we use ngrok to launch a 
server on the corresponding port:
```
ngrok http --subdomain=whisper-jax 8000
```

We can now send json requests to our endpoint using ngrok. The function `transcribe_audio` loads an audio file, encodes it 
in bytes, sends it to our endpoint, and returns the transcription:

```python
import base64
from transformers.pipelines.audio_utils import ffmpeg_read
import requests

API_URL = "https://whisper-jax.ngrok.io/generate/"  # make sure this URL matches your ngrok subdomain


def query(payload):
    """Send json payload to ngrok API URL and return response."""
    response = requests.post(API_URL, json=payload)
    return response.json(), response.status_code


def transcribe_audio(audio_file, task="transcribe", return_timestamps=False):
    with open(audio_file, "rb") as f:
        inputs = f.read()
    inputs = ffmpeg_read(inputs, sampling_rate=16000)
    # encode to bytes to make json compatible
    inputs = {"array": base64.b64encode(inputs.tobytes()).decode(), "sampling_rate": 16000}
    # format as a json payload and send query
    payload = {"inputs": inputs, "task": task, "return_timestamps": return_timestamps}
    data, status_code = query(payload)

    if status_code == 200:
        output = {"text": data["text"], "chunks": data.get("chunks", None)}
    else:
        output = data["detail"]

    return output

# transcribe an audio file using our endpoint
output = transcribe_audio("audio.mp3")
```

Finally, we can create a Gradio demo for the frontend, the code for which resides in [`app.py`](app/app.py). You can launch this 
application by providing the ngrok subdomain:
```
API_URL=https://whisper-jax.ngrok.io/generate/ API_URL_FROM_FEATURES=https://whisper-jax.ngrok.io/generate_from_features/ python app.py
```

This will launch a Gradio demo with the same interface as the official Whisper JAX demo.

[^1]: See WER results from Colab: https://colab.research.google.com/drive/1rS1L4YSJqKUH_3YxIQHBI982zso23wor?usp=sharing