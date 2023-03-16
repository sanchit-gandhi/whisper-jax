import math
from flax.core.frozen_dict import freeze
import jax.numpy as jnp
import jax
from jax.sharding import PartitionSpec as P

from transformers import WhisperProcessor
from .modeling_flax_whisper import FlaxWhisperForConditionalGeneration
from .partitioner import PjitPartitioner
from .train_state import InferenceState

import numpy as np

from transformers.utils import logging

jax.config.update("jax_array", True)

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

    def chunk_iter_with_batch(self, inputs, chunk_len, stride_left, stride_right, batch_size):
        inputs_len = inputs.shape[0]
        step = chunk_len - stride_left - stride_right

        all_chunk_start_idx = np.arange(0, inputs_len, step)
        num_samples = len(all_chunk_start_idx)

        num_batches = math.ceil(num_samples / batch_size)
        batch_idx = np.array_split(np.arange(num_samples), num_batches)

        for i, idx in enumerate(batch_idx):
            chunk_start_idx = all_chunk_start_idx[idx]

            chunk_end_idx = chunk_start_idx + chunk_len

            chunks = [inputs[chunk_start:chunk_end] for chunk_start, chunk_end in zip(chunk_start_idx, chunk_end_idx)]
            processed = self.feature_extractor(chunks, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="np")

            _stride_left = np.where(chunk_start_idx == 0, 0, stride_left)
            is_last = np.where(stride_right > 0, chunk_end_idx > inputs_len, chunk_end_idx >= inputs_len)
            _stride_right = np.where(is_last, 0, stride_right)

            chunk_lens = [chunk.shape[0] for chunk in chunks]
            strides = [(chunk_l, _stride_l, _stride_r) for chunk_l, _stride_l, _stride_r in zip(chunk_lens, _stride_left, _stride_right)]

            yield {"stride": strides, **processed}

    def preprocess_batch(self, inputs, chunk_length_s=0, stride_length_s=None, batch_size=None):
        # TODO(SG): handle more generic input types, currently assume inputs is a dict {"array":, "sampling_rate":}
        array = inputs.get("array")
        in_sampling_rate = inputs.get("sampling_rate")
        stride = inputs.get("stride", None)

        if in_sampling_rate != self.feature_extractor.sampling_rate:
            try:
                import librosa
                import soundfile as sf
            except ImportError as err:
                raise ImportError("To support resampling audio files, please install 'librosa' and 'soundfile'.") from err

            array = librosa.resample(array, orig_sr=in_sampling_rate, target_sr=self.feature_extractor.sampling_rate)
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

            for item in self.chunk_iter_with_batch(
                    array, chunk_len, stride_left, stride_right, batch_size,
            ):
                yield item
        else:
            processed = self.feature_extractor(
                array, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="np"
            )
            if stride is not None:
                processed["stride"] = stride
            yield processed

    def postprocess(self, model_outputs, return_timestamps=None, return_language=None):
        # unpack the outputs from list(dict(list)) to list(dict)
        model_outputs = [dict(zip(output, t)) for output in model_outputs for t in zip(*output.values())]

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
        # We need to keep track of some additional input arguments for post-processing so need to forward these on after running generation
        if generate_kwargs is None:
            generate_kwargs = {}

        if return_timestamps:
            generate_kwargs["return_timestamps"] = return_timestamps

        input_features = model_inputs.pop("input_features")

        input_batch_size = input_features.shape[0]
        # TODO(SG): handle variable batch lengths
        if input_batch_size != batch_size:
            padding = np.zeros([batch_size - input_batch_size, *input_features.shape[1:]], input_features.dtype)
            input_features = np.concatenate([input_features, padding])

        pred_ids = self.generate(input_features)[:input_batch_size]

        # tokenizer's decode method expects an extra dim - we insert it here for convenience
        out = {"tokens": np.asarray(pred_ids[:, None, :])}

        stride = model_inputs.pop("stride", None)
        if stride is not None:
            out["stride"] = stride

        return out

    def __call__(self, inputs, chunk_length_s=30, stride_length_s=None, return_timestamps=None, return_language=None, generate_kwargs=None, batch_size=None):
        dataloader = self.preprocess_batch(inputs, chunk_length_s=chunk_length_s, stride_length_s=stride_length_s, batch_size=batch_size)

        model_outputs = []
        # iterate over our chunked audio samples
        for batch in dataloader:
            model_outputs.append(self.forward(batch, batch_size=batch_size))

        post_processed = self.postprocess(model_outputs, return_timestamps=return_timestamps)
        return post_processed
