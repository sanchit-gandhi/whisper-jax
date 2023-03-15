import argparse
import time
from datasets import load_dataset, concatenate_datasets
from flax.core.frozen_dict import freeze
import jax.numpy as jnp
import jax
from jax.experimental import PartitionSpec as P
from transformers import WhisperProcessor, WhisperConfig

from whisper_jax import FlaxWhisperForConditionalGeneration, PjitPartitioner, InferenceState

import datasets
datasets.logging.set_verbosity(datasets.logging.CRITICAL)

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Whisper large-v2")
    parser.add_argument(
        "--model_parallel_submesh",
        type=int,
        nargs='+',
        default="2, 2, 1, 1",
        help="Model parallel submesh.",
    )
    parser.add_argument(
        "--use_palm",
        action="store_true",
        help="Whether to use PALM layout.",
    )
    parser.add_argument(
        "--use_whisper",
        action="store_true",
        help="Whether to use Whipser layout.",
    )
    args = parser.parse_args()
    return args


# 2D parameter and activation partitioning
logical_axis_rules_full = [
    ("batch", "data"),
    ("mlp", "model"),
    ("heads", "model"),
    ("vocab", None),
    # shard both activations and weight matrices on the remaining available axis
    ("embed", "model"),
    ("embed", "data"),
    ("joined_kv", None),
    ("kv", None),
    ("length", None),
    ("num_mel", None),
    ("channels", None)
]

# 2D parameter and activation partitioning from PALM
logical_axis_rules_palm = [
    ("batch", None),
    ("mlp", "data"),
    ("heads", "data"),
    ("vocab", None),
    ("embed", "model"),
    ("embed", "model"),
    ("joined_kv", None),
    ("kv", None),
    ("length", None),
    ("num_mel", None),
    ("channels", None)
]

# 2D parameter and activation partitioning from PALM
logical_axis_rules_whisper = [
    ("batch", None),
    ("mlp", "data"),
    ("heads", "data"),
    ("vocab", None),
    ("embed", "model"),
    ("embed", "model"),
    ("joined_kv", None),
    ("kv", None),
    ("length", "model"),
    ("num_mel", None),
    ("channels", None)
]

BATCH_SIZES = [4, 8, 16, 32]
NUM_BATCHES = 100
NUM_TOKENS = 25
CHECKPOINT = "large-v2"

def main():
    args = parse_args()
    print(args.model_parallel_submesh)
    print(args.use_palm)
    # processors/tokenizers are the same for all models, so just load from tiny and preprocess once
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

    def preprocess(batch):
        batch["input_features"] = processor(
            batch["audio"]["array"], sampling_rate=16000, return_tensors="np"
        ).input_features[0]
        return batch

    librispeech = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    dataset_processed = librispeech.map(preprocess, remove_columns=librispeech.column_names)

    config = WhisperConfig.from_pretrained(f"openai/whisper-{CHECKPOINT}")
    model = FlaxWhisperForConditionalGeneration(config, _do_init=False, dtype=jnp.bfloat16)
    # to init the params
    params = model.init_weights(model.key, model.input_shape)

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

    logical_axis_rules = logical_axis_rules_full
    if args.use_palm:
        logical_axis_rules = logical_axis_rules_palm
    if args.use_whisper:
        logical_axis_rules = logical_axis_rules_whisper

    partitioner = PjitPartitioner(
        model_parallel_submesh=tuple(args.model_parallel_submesh),
        logical_axis_rules=logical_axis_rules,
    )

    mesh_axes = partitioner.get_mesh_axes(state)
    params_spec = mesh_axes.params

    p_shard_params = partitioner.partition(model.to_bf16, (params_spec,), params_spec)

    def generate(params, input_features):
        output_ids = model.generate(input_features, params=params, max_new_tokens=25).sequences
        return output_ids

    p_generate = partitioner.partition(
        generate,
        in_axis_resources=(params_spec, P("data")),
        out_axis_resources=P("data"),
    )

    # This will auto-magically run in mesh context
    params = p_shard_params(freeze(params))

    for batch_size in BATCH_SIZES:
        eval_dataset = dataset_processed.select(range(batch_size // 2))
        eval_dataset = concatenate_datasets([eval_dataset for _ in range(2 * NUM_BATCHES)])

        eval_dataloader = eval_dataset.with_format("numpy").iter(batch_size=batch_size)

        # warm-up step
        batch = next(iter(eval_dataloader))
        pred_ids = p_generate(freeze(params), batch["input_features"])

        start = time.time()
        for batch in eval_dataloader:
            pred_ids = p_generate(freeze(params), batch["input_features"])
        runtime = time.time() - start

        print(f"{batch_size}: {runtime:.06}")

if __name__ == "__main__":
    main()