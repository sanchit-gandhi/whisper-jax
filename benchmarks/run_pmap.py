import time

import jax
import jax.numpy as jnp
from datasets import concatenate_datasets, load_dataset
from flax import jax_utils
from flax.training.common_utils import shard
from transformers import FlaxWhisperForConditionalGeneration, WhisperProcessor


BATCH_SIZES = [4, 8, 16, 32, 64, 128]
NUM_BATCHES = 100
NUM_TOKENS = 25

model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v2",
    _do_init=False,
    dtype=jnp.bfloat16,
)

params = jax_utils.replicate(params)


def generate_fn(batch):
    pred_ids = model.generate(batch, params=params, max_new_tokens=NUM_TOKENS, min_new_tokens=NUM_TOKENS)
    return pred_ids.sequences


p_generate_fn = jax.pmap(generate_fn, "batch")

# processors/tokenizers are the same for all models, so just load from tiny and preprocess once
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")


def preprocess(batch):
    batch["input_features"] = processor(
        batch["audio"]["array"], sampling_rate=16000, return_tensors="np"
    ).input_features[0]
    return batch


# load a dataset of 73 audio samples
librispeech = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset_processed = librispeech.map(preprocess, remove_columns=librispeech.column_names)

for batch_size in BATCH_SIZES:
    eval_dataset = dataset_processed.select(range(batch_size // 2))
    eval_dataset = concatenate_datasets([eval_dataset for _ in range(2 * NUM_BATCHES)])

    eval_dataloader = eval_dataset.with_format("numpy").iter(batch_size=batch_size)

    # warm-up step
    batch = next(iter(eval_dataloader))
    input_features = shard(batch["input_features"])
    pred_ids = p_generate_fn(input_features)

    start = time.time()
    for batch in eval_dataloader:
        input_features = shard(batch["input_features"])
        pred_ids = p_generate_fn(input_features)
    runtime = time.time() - start

    print(f"{batch_size}: {runtime:.06}")
