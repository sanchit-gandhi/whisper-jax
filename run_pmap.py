from transformers import FlaxWhisperForConditionalGeneration
import jax.numpy as jnp
import numpy as np
import jax
from flax.training.common_utils import shard

import time

from jax.experimental.compilation_cache import compilation_cache as cc

cc.initialize_cache("./jax_cache")

BATCH_SIZES = [4, 8, 16, 32, 64, 128, 256]
NUM_BATCHES = 100
NUM_TOKENS = 25

model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small.en",
    _do_init=False,
    dtype=jnp.bfloat16,
)

def generate_fn(batch):
    pred_ids = model.generate(batch, params=params, max_new_tokens=NUM_TOKENS, min_new_tokens=NUM_TOKENS)
    return pred_ids.sequences

p_generate_fn = jax.pmap(generate_fn, "batch")

for batch_size in BATCH_SIZES:
    # keep inputs on host for async dispatch
    input_features = np.ones((batch_size, 80, 3000))
    input_features = shard(input_features)

    # warm-up
    out = p_generate_fn(input_features)

    # generate
    start = time.time()
    for i in range(NUM_BATCHES):
        out = p_generate_fn(input_features)
    runtime = time.time() - start

    print(f"{batch_size}: {runtime:.06}")
