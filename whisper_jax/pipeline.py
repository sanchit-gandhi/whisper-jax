from flax.core.frozen_dict import freeze
import jax.numpy as jnp
import jax
from jax.sharding import PartitionSpec as P
from jax.experimental.compilation_cache import compilation_cache as cc
from torch.utils.data import DataLoader

from transformers import WhisperProcessor
from transformers.pipelines.base import no_collate_fn
from transformers.pipelines.pt_utils import PipelineIterator, PipelineChunkIterator, PipelinePackIterator
from .modeling_flax_whisper import FlaxWhisperForConditionalGeneration
from .partitioner import PjitPartitioner
from .train_state import InferenceState

import numpy as np

from transformers.utils import logging

jax.config.update("jax_array", True)
cc.initialize_cache("./jax_cache")

logger = logging.get_logger(__name__)

# 2D parameter and activation partitioning from PALM
logical_axis_rules_palm = (
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
)


def chunk_iter(inputs, feature_extractor, chunk_len, stride_left, stride_right):
    inputs_len = inputs.shape[0]
    step = chunk_len - stride_left - stride_right
    for chunk_start_idx in range(0, inputs_len, step):
        chunk_end_idx = chunk_start_idx + chunk_len
        chunk = inputs[chunk_start_idx:chunk_end_idx]
        processed = feature_extractor(chunk, sampling_rate=feature_extractor.sampling_rate, return_tensors="np")
        _stride_left = 0 if chunk_start_idx == 0 else stride_left
        # all right strides must be full, otherwise it is the last item
        is_last = chunk_end_idx > inputs_len if stride_right > 0 else chunk_end_idx >= inputs_len
        _stride_right = 0 if is_last else stride_right

        chunk_len = chunk.shape[0]
        stride = (chunk_len, _stride_left, _stride_right)
        if chunk.shape[0] > _stride_left:
            yield {"is_last": is_last, "stride": stride, **processed}
        if is_last:
            break


class FlaxWhisperPipline:
    def __init__(
        self,
        checkpoint="openai/whisper-large-v2",
        dtype=jnp.bfloat16,
        model_parallel_submesh=(1, 1, 1, 1),
        logical_axis_rules=logical_axis_rules_palm,
        max_length=None,
    ):
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.model_parallel_submesh = model_parallel_submesh
        self.logical_axis_rules = logical_axis_rules

        self.processor = WhisperProcessor.from_pretrained(self.checkpoint)
        self.feature_extractor = self.processor.feature_extractor
        self.tokenizer = self.processor.tokenizer

        self.model, self.params = FlaxWhisperForConditionalGeneration.from_pretrained(
            self.checkpoint,
            _do_init=False,
            dtype=self.dtype,
        )

        self.max_length = max_length if max_length is not None else self.model.generation_config.max_length

        def init_fn():
            input_shape = (1, 80, 3000)

            input_features = jnp.zeros(input_shape, dtype="f4")
            input_features = input_features.at[(..., -1)].set(self.model.config.eos_token_id)

            decoder_input_ids = jnp.zeros((input_shape[0], 1), dtype="i4")
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)

            batch_size, sequence_length = decoder_input_ids.shape
            decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

            rng = jax.random.PRNGKey(0)
            init_params = self.model.module.init(
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
            params=freeze(self.model.params_shape_tree),
            params_axes=freeze(param_axes),
            flax_mutables=None,
            flax_mutables_axes=param_axes,
        )

        partitioner = PjitPartitioner(
            model_parallel_submesh=self.model_parallel_submesh,
            logical_axis_rules=self.logical_axis_rules
        )

        mesh_axes = partitioner.get_mesh_axes(state)
        params_spec = mesh_axes.params

        self.p_shard_params = partitioner.partition(self.model.to_bf16, (params_spec,), params_spec)

        def generate(params, input_features):
            # TODO(SG): add task and language (static argnums?)
            output_ids = self.model.generate(input_features, params=params, max_length=self.max_length)
            return output_ids

        self.p_generate = partitioner.partition(
            generate,
            in_axis_resources=(params_spec, P("data")),
            out_axis_resources=P("data"),
        )

    def shard_params(self):
        # This will auto-magically run in mesh context
        self.params = self.p_shard_params(freeze(self.params))

    def generate(self, input_features):
        output_ids = self.p_generate(freeze(self.params), input_features).sequences
        return output_ids

    def preprocess(self, inputs, chunk_length_s=0, stride_length_s=None):
        # TODO(SG): handle more generic input types, currently assume inputs is a dict {"array":, "sampling_rate":}
        array = inputs.get("array")
        in_sampling_rate = inputs.get("sampling_rate")
        stride = inputs.get("stride", None)

        if in_sampling_rate != self.feature_extractor.sampling_rate:
            # TODO(SG): resampling ...
            ratio = self.feature_extractor.sampling_rate / in_sampling_rate
        else:
            ratio = 1

        if stride is not None:
            if stride[0] + stride[1] > inputs.shape[0]:
                raise ValueError("Stride is too large for input")

            # Stride needs to get the chunk length here, it's going to get
            # swallowed by the `feature_extractor` later, and then batching
            # can add extra data in the inputs, so we need to keep track
            # of the original length in the stride so we can cut properly.
            stride = (inputs.shape[0], int(round(stride[0] * ratio)), int(round(stride[1] * ratio)))

        if chunk_length_s:
            if stride_length_s is None:
                stride_length_s = chunk_length_s / 6

            if isinstance(stride_length_s, (int, float)):
                stride_length_s = [stride_length_s, stride_length_s]

            chunk_len = round(chunk_length_s * self.feature_extractor.sampling_rate)
            stride_left = round(stride_length_s[0] * self.feature_extractor.sampling_rate)
            stride_right = round(stride_length_s[1] * self.feature_extractor.sampling_rate)

            if chunk_len < stride_left + stride_right:
                raise ValueError("Chunk length must be superior to stride length")

            for item in chunk_iter(
                array, self.feature_extractor, chunk_len, stride_left, stride_right,
            ):
                yield item
        else:
            processed = self.feature_extractor(
                array, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="np"
            )
            if stride is not None:
                processed["stride"] = stride
            yield {"is_last": True, **processed}

    def postprocess(
        self, model_outputs, return_timestamps=None, return_language=None
    ):
        # Optional return types
        final_items = []

        for outputs in model_outputs:
            items = np.array(outputs["tokens"])
            final_items.append(items)

        time_precision = self.feature_extractor.chunk_length / self.model.config.max_source_positions
        # Send the chunking back to seconds, it's easier to handle in whisper
        sampling_rate = self.feature_extractor.sampling_rate
        for output in model_outputs:
            if "stride" in output:
                chunk_len, stride_left, stride_right = output["stride"]
                # Go back in seconds
                chunk_len /= sampling_rate
                stride_left /= sampling_rate
                stride_right /= sampling_rate
                output["stride"] = chunk_len, stride_left, stride_right

        text, optional = self.tokenizer._decode_asr(
            model_outputs,
            return_timestamps=return_timestamps,
            return_language=return_language,
            time_precision=time_precision,
        )
        return {"text": text, **optional}

    def forward(self, model_inputs, batch_size=None, return_timestamps=False, generate_kwargs=None):
        if generate_kwargs is None:
            generate_kwargs = {}

        if return_timestamps:
            generate_kwargs["return_timestamps"] = return_timestamps
        is_last = model_inputs.pop("is_last")

        input_features = model_inputs.pop("input_features")

        input_batch_size = input_features.shape[0]
        # TODO(SG): handle variable batch lengths
        if input_batch_size != batch_size:
            padding = np.zeros([batch_size - input_batch_size, *input_features.shape[1:]], input_features.dtype)
            input_features = np.concatenate([input_features, padding])

        pred_ids = self.generate(input_features)[:input_batch_size]
        out = {"tokens": np.asarray(pred_ids)}

        stride = model_inputs.pop("stride", None)
        if stride is not None:
            out["stride"] = stride

        return {"is_last": is_last, **out}

    def __call__(self, inputs, chunk_length_s=30, stride_length_s=None, return_timestamps=None, return_language=None, generate_kwargs=None, batch_size=4, num_workers=1):
        dataset = PipelineChunkIterator([inputs], self.preprocess, {"chunk_length_s": chunk_length_s, "stride_length_s": stride_length_s})
        collate_fn = no_collate_fn if batch_size == 1 else pad_collate_fn()
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn, drop_last=False)
        model_iterator = PipelinePackIterator(dataloader, self.forward, {"batch_size": batch_size, "return_timestamps": return_timestamps}, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, {})
        return next(iter(final_iterator))

def _pad(items, key):
    if isinstance(items[0][key], np.ndarray):
        if key == "input_features":
            # this is probably a mel spectrogram batched
            return np.concatenate([item[key] for item in items], axis=0)
    else:
        return [item[key] for item in items]


def pad_collate_fn():
    def inner(items):
        keys = set(items[0].keys())
        for item in items:
            if set(item.keys()) != keys:
                raise ValueError(
                    f"The elements of the batch contain different keys. Cannot batch them ({set(item.keys())} !="
                    f" {keys})"
                )
        # input_values, input_pixels, input_ids, ...
        padded = {}
        for key in keys:
            padded[key] = _pad(items, key)
        return padded
    return inner
