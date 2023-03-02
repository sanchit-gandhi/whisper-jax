from tensorflow.python._pywrap_tensorflow_internal import *
from modeling_flax_whisper import FlaxWhisperForConditionalGeneration
from flax.core.frozen_dict import freeze
import jax.numpy as jnp
import jax
from jax.experimental import PartitionSpec as P

from partitioner import PjitPartitioner
from train_state import InferenceState

num_mp_partitions = 1

# 2D parameter and activation partitioning
# TODO: update for device
logical_axis_rules_tpu = [
    ('batch', 'data'),
    ('mlp', 'model'),
    ('heads', 'model'),
    ('vocab', 'model'),
    # shard both activations and weight matrices on the remaining available axis
    ('embed', 'model'),
    ('embed', 'data'),
    ('kv', None),
    ('joined_kv', None),
    ('relpos_buckets', None),
    ('abspos_buckets', None),
    ('length', None),
    ('layers', None),
    ('stack', None),
    ('mlp_activations', None),
]

logical_axis_rules_full = [
    ('batch', 'data'),
    ('mlp', 'model'),
    ('heads', 'model'),
    ('vocab', 'model'),
    # shard both activations and weight matrices on the remaining available axis
    ('embed', 'model'),
    ('embed', 'data'),
    ('kv', None),
    ('joined_kv', None),
    ('relpos_buckets', None),
    ('abspos_buckets', None),
    ('length', None),
    ('layers', None),
    ('stack', None),
    ('mlp_activations', None),
]

model, params = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", _do_init=False)

def init_fn():
    input_shape = (1, 80, 3000)

    input_features = jnp.zeros(input_shape, dtype="f4")
    input_features = input_features.at[(..., -1)].set(model.config.eos_token_id)

    decoder_input_ids = jnp.zeros((input_shape[0], 1), dtype="i4")
    decoder_attention_mask = jnp.ones_like(decoder_input_ids)

    batch_size, sequence_length = decoder_input_ids.shape
    decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

    rng = jax.random.PRNGKey(0)
    return model.module.init(rng, input_features=input_features, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, decoder_position_ids=decoder_position_ids, return_dict=False)

# Axis names metadata
param_axes = jax.eval_shape(init_fn)["params_axes"]

# create InferenceState, since the partitioner expects it.
state = InferenceState(
    step=jnp.array(0),
    params=freeze(model.params_shape_tree),
    params_axes=freeze(param_axes),
    flax_mutables=None,
    flax_mutables_axes=param_axes,
)

partitioner = PjitPartitioner(num_mp_partitions, logical_axis_rules=logical_axis_rules_full)

mesh_axes = partitioner.get_mesh_axes(state)
params_spec = mesh_axes.params

# TODO: to bf16 for TPU
p_shard_params = partitioner.partition(model.to_fp16, (params_spec,), params_spec)

def generate(params, input_features, attention_mask=None):
    output_ids = model.generate(input_features, attention_mask=attention_mask, params=params).sequences
    return output_ids


p_generate = partitioner.partition(
    generate,
    in_axis_resources=(params_spec, P("data"), P("data")),
    out_axis_resources=P("data")
)

# This will auto-magically run in mesh context
params = p_shard_params(freeze(params))

inputs = jnp.ones(1, 80, 3000, dtype=jnp.float16)

gen_ids = p_generate(freeze(params), inputs["input_features"])
