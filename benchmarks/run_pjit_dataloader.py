import time

import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from flax.core.frozen_dict import freeze
from jax.experimental.compilation_cache import compilation_cache as cc
from jax.sharding import PartitionSpec as P
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperProcessor

from whisper_jax import FlaxWhisperForConditionalGeneration, InferenceState, PjitPartitioner


cc.initialize_cache("./jax_cache")
jax.config.update("jax_array", True)

BATCH_SIZE = 128
NUM_TOKENS = 256
NUM_BATCHES = 5
NUM_WORKERS = 4
CHECKPOINT = "large-v2"
STREAMING = True

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
    f"openai/whisper-{CHECKPOINT}",
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

partitioner = PjitPartitioner(
    num_partitions=1,
    logical_axis_rules=logical_axis_rules_dp,
)

mesh_axes = partitioner.get_mesh_axes(state)
params_spec = mesh_axes.params

p_shard_params = partitioner.partition(model.to_bf16, (params_spec,), params_spec)


def generate(params, input_features):
    output_ids = model.generate(input_features, params=params, max_length=NUM_TOKENS).sequences
    return output_ids


p_generate = partitioner.partition(
    generate,
    in_axis_resources=(params_spec, P("data")),
    out_axis_resources=P("data"),
)

# This will auto-magically run in mesh context
params = p_shard_params(freeze(params))

# warm-up
pred_ids = p_generate(freeze(params), np.ones((BATCH_SIZE, 80, 3000)))

# processors/tokenizers are the same for all models, so just load from tiny and preprocess once
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")


def preprocess(batch):
    batch["input_features"] = processor(
        batch["audio"]["array"], sampling_rate=16000, return_tensors="np"
    ).input_features[0]
    return batch


librispeech = load_dataset("speechcolab/gigaspeech", "l", split="train", streaming=STREAMING, use_auth_token=True)
librispeech_features = librispeech.features.keys()

librispeech_processed = librispeech.map(preprocess, remove_columns=librispeech_features)

eval_dataloader = DataLoader(
    librispeech_processed, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, persistent_workers=True
)

all_load_times = 0
all_runtimes = 0

load_start = time.time()
for i, batch in tqdm(enumerate(eval_dataloader)):
    input_features = np.asarray(batch.pop("input_features"))
    load_time = time.time() - load_start
    all_load_times += load_time

    input_batch_size = input_features.shape[0]

    if input_batch_size != BATCH_SIZE:
        padding = np.zeros([BATCH_SIZE - input_batch_size, *input_features.shape[1:]], input_features.dtype)
        input_features = np.concatenate([input_features, padding])

    generate_start = time.time()
    pred_ids = p_generate(freeze(params), input_features)[:input_batch_size]
    generate_runtime = time.time() - generate_start
    all_runtimes += generate_runtime

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

    if i + 1 == NUM_BATCHES:
        break

print(all_load_times)
print(all_runtimes)
