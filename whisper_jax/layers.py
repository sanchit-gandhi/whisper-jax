# coding=utf-8
# Copyright 2023 The T5X Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dense attention classes and mask/weighting functions."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic

import dataclasses
import functools
import operator
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.linen.dtypes import promote_dtype
from jax import lax, random


# from flax.linen.partitioning import param_with_axes, with_sharding_constraint
param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint


# Type annotations
Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision, lax.Precision]]
DotGeneralT = Callable[..., Array]
ConvGeneralDilatedT = Callable[..., Array]
PaddingLike = Union[str, int, Sequence[Union[int, Tuple[int, int]]]]
LaxPadding = Union[str, Sequence[Tuple[int, int]]]

# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]
InitializerAxis = Union[int, Tuple[int, ...]]
NdInitializer = Callable[[PRNGKey, Shape, DType, InitializerAxis, InitializerAxis], Array]

default_embed_init = nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)


# ------------------------------------------------------------------------------
# Temporary inlined JAX N-d initializer code
# TODO(levskaya): remove once new JAX release is out.
# ------------------------------------------------------------------------------
def _compute_fans(shape: jax.core.DShapedArray, in_axis=-2, out_axis=-1):
    """Inlined JAX `nn.initializer._compute_fans`."""
    if isinstance(in_axis, int):
        in_size = shape[in_axis]
    else:
        in_size = int(np.prod([shape[i] for i in in_axis]))
    if isinstance(out_axis, int):
        out_size = shape[out_axis]
    else:
        out_size = int(np.prod([shape[i] for i in out_axis]))
    receptive_field_size = shape.total / in_size / out_size
    fan_in = in_size * receptive_field_size
    fan_out = out_size * receptive_field_size
    return fan_in, fan_out


def variance_scaling(scale, mode, distribution, in_axis=-2, out_axis=-1, dtype=jnp.float_):
    """Inlined JAX `nn.initializer.variance_scaling`."""

    def init(key, shape, dtype=dtype):
        return jnp.zeros(shape, dtype=dtype)
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        shape = jax.core.as_named_shape(shape)
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError("invalid mode for variance scaling initializer: {}".format(mode))
        variance = jnp.array(scale / denominator, dtype=dtype)

        if distribution == "truncated_normal":
            # constant is stddev of standard normal truncated to (-2, 2)
            stddev = jnp.sqrt(variance) / jnp.array(0.87962566103423978, dtype)
            return random.truncated_normal(key, -2, 2, shape, dtype) * stddev
        elif distribution == "normal":
            return random.normal(key, shape, dtype) * jnp.sqrt(variance)
        elif distribution == "uniform":
            return random.uniform(key, shape, dtype, -1) * jnp.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling " "initializer: {}".format(distribution))

    return init


# ------------------------------------------------------------------------------


def nd_dense_init(scale, mode, distribution):
    """Initializer with in_axis, out_axis set at call time."""

    def init_fn(key, shape, dtype, in_axis, out_axis):
        fn = variance_scaling(scale, mode, distribution, in_axis, out_axis)
        return fn(key, shape, dtype)

    return init_fn


def dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: DType = jnp.float32,
    float32_logits: bool = False,
):
    """Computes dot-product attention given query, key, and value.

    This is the core function for applying attention based on
    https://arxiv.org/abs/1706.03762. It calculates the attention weights given
    query and key and combines the values using the attention weights.

    Args:
      query: queries for calculating attention with shape of `[batch, q_length,
        num_heads, qk_depth_per_head]`.
      key: keys for calculating attention with shape of `[batch, kv_length,
        num_heads, qk_depth_per_head]`.
      value: values to be used in attention with shape of `[batch, kv_length,
        num_heads, v_depth_per_head]`.
      bias: bias for the attention weights. This should be broadcastable to the
        shape `[batch, num_heads, q_length, kv_length]` This can be used for
        incorporating causal masks, padding masks, proximity bias, etc.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      dtype: the dtype of the computation (default: float32)
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.

    Returns:
      Output of shape `[batch, length, num_heads, v_depth_per_head]`.
    """
    assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], "q, k, v batch dims must match."
    assert query.shape[-2] == key.shape[-2] == value.shape[-2], "q, k, v num_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

    # Casting logits and softmax computation for float32 for model stability.
    if float32_logits:
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)

    # `attn_weights`: [batch, num_heads, q_length, kv_length]
    attn_weights = jnp.einsum("bqhd,bkhd->bhqk", query, key)

    # Apply attention bias: masking, dropout, proximity bias, etc.
    if bias is not None:
        attn_weights = attn_weights + bias.astype(attn_weights.dtype)

    # Normalize the attention weights across `kv_length` dimension.
    attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

    # Apply attention dropout.
    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        # T5 broadcasts along the "length" dim, but unclear which one that
        # corresponds to in positional dimensions here, assuming query dim.
        dropout_shape = list(attn_weights.shape)
        dropout_shape[-2] = 1
        keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        keep = jnp.broadcast_to(keep, attn_weights.shape)
        multiplier = keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype)
        attn_weights = attn_weights * multiplier

    # Take the linear combination of `value`.
    return jnp.einsum("bhqk,bkhd->bqhd", attn_weights, value)


dynamic_vector_slice_in_dim = jax.vmap(lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))


class MultiHeadDotProductAttention(nn.Module):
    """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      head_dim: dimension of each head.
      dtype: the dtype of the computation.
      dropout_rate: dropout rate
      kernel_init: initializer for the kernel of the Dense layers.
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.
    """

    num_heads: int
    head_dim: int
    dtype: DType = jnp.float32
    dropout_rate: float = 0.0
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal")
    float32_logits: bool = False  # computes logits in float32 for stability.

    @nn.compact
    def __call__(
        self,
        inputs_q: Array,
        inputs_kv: Array,
        mask: Optional[Array] = None,
        bias: Optional[Array] = None,
        *,
        decode: bool = False,
        deterministic: bool = False,
    ) -> Array:
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        There are two modes: decoding and non-decoding (e.g., training). The mode is
        determined by `decode` argument. For decoding, this method is called twice,
        first to initialize the cache and then for an actual decoding process. The
        two calls are differentiated by the presence of 'cached_key' in the variable
        dict. In the cache initialization stage, the cache variables are initialized
        as zeros and will be filled in the subsequent decoding process.

        In the cache initialization call, `inputs_q` has a shape [batch, length,
        q_features] and `inputs_kv`: [batch, length, kv_features]. During the
        incremental decoding stage, query, key and value all have the shape [batch,
        1, qkv_features] corresponding to a single step.

        Args:
          inputs_q: input queries of shape `[batch, q_length, q_features]`.
          inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
          mask: attention mask of shape `[batch, num_heads, q_length, kv_length]`.
          bias: attention bias of shape `[batch, num_heads, q_length, kv_length]`.
          decode: Whether to prepare and use an autoregressive cache.
          deterministic: Disables dropout if set to True.

        Returns:
          output of shape `[batch, length, q_features]`.
        """
        projection = functools.partial(
            DenseGeneral,
            axis=-1,
            features=(self.num_heads, self.head_dim),
            kernel_axes=("embed", "heads", "kv"),
            dtype=self.dtype,
        )

        # NOTE: T5 does not explicitly rescale the attention logits by
        #       1/sqrt(depth_kq)!  This is folded into the initializers of the
        #       linear transformations, which is equivalent under Adafactor.
        depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)

        def query_init(*args):
            return self.kernel_init(*args) / depth_scaling

        # Project inputs_q to multi-headed q/k/v
        # dimensions are then [batch, length, num_heads, head_dim]
        query = projection(kernel_init=query_init, name="query")(inputs_q)
        key = projection(kernel_init=self.kernel_init, name="key")(inputs_kv)
        value = projection(kernel_init=self.kernel_init, name="value")(inputs_kv)

        query = with_sharding_constraint(query, ("batch", "length", "heads", "kv"))
        key = with_sharding_constraint(key, ("batch", "length", "heads", "kv"))
        value = with_sharding_constraint(value, ("batch", "length", "heads", "kv"))

        if decode:
            # Detect if we're initializing by absence of existing cache data.
            is_initialized = self.has_variable("cache", "cached_key")

            # The key and value have dimension [batch, length, num_heads, head_dim],
            # but we cache them as [batch, num_heads, head_dim, length] as a TPU
            # fusion optimization. This also enables the "scatter via one-hot
            # broadcast" trick, which means we do a one-hot broadcast instead of a
            # scatter/gather operations, resulting in a 3-4x speedup in practice.
            def swap_dims(x):
                return x[:-3] + tuple(x[i] for i in [-2, -1, -3])

            cached_key = self.variable("cache", "cached_key", jnp.zeros, swap_dims(key.shape), key.dtype)
            cached_value = self.variable("cache", "cached_value", jnp.zeros, swap_dims(value.shape), value.dtype)
            cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))
            if is_initialized:
                batch, num_heads, head_dim, length = cached_key.value.shape
                # During fast autoregressive decoding, we feed one position at a time,
                # and cache the keys and values step by step.
                # Sanity shape check of cached key against input query.
                expected_shape = (batch, 1, num_heads, head_dim)
                if expected_shape != query.shape:
                    raise ValueError(
                        "Autoregressive cache shape error, "
                        "expected query shape %s instead got %s." % (expected_shape, query.shape)
                    )

                # Create a OHE of the current index. NOTE: the index is increased below.
                cur_index = cache_index.value
                one_hot_indices = jax.nn.one_hot(cur_index, length, dtype=key.dtype)
                # In order to update the key, value caches with the current key and
                # value, we move the length axis to the back, similar to what we did for
                # the cached ones above.
                # Note these are currently the key and value of a single position, since
                # we feed one position at a time.
                one_token_key = jnp.moveaxis(key, -3, -1)
                one_token_value = jnp.moveaxis(value, -3, -1)
                # Update key, value caches with our new 1d spatial slices.
                # We implement an efficient scatter into the cache via one-hot
                # broadcast and addition.
                key = cached_key.value + one_token_key * one_hot_indices
                value = cached_value.value + one_token_value * one_hot_indices
                cached_key.value = key
                cached_value.value = value
                cache_index.value = cache_index.value + 1
                # Move the keys and values back to their original shapes.
                key = jnp.moveaxis(key, -1, -3)
                value = jnp.moveaxis(value, -1, -3)

                # Causal mask for cached decoder self-attention: our single query
                # position should only attend to those key positions that have already
                # been generated and cached, not the remaining zero elements.
                mask = combine_masks(
                    mask,
                    jnp.broadcast_to(
                        jnp.arange(length) <= cur_index,
                        # (1, 1, length) represent (head dim, query length, key length)
                        # query length is 1 because during decoding we deal with one
                        # index.
                        # The same mask is applied to all batch elements and heads.
                        (batch, 1, 1, length),
                    ),
                )

                # Grab the correct relative attention bias during decoding. This is
                # only required during single step decoding.
                if bias is not None:
                    # The bias is a full attention matrix, but during decoding we only
                    # have to take a slice of it.
                    # This is equivalent to bias[..., cur_index:cur_index+1, :].
                    bias = dynamic_vector_slice_in_dim(jnp.squeeze(bias, axis=0), jnp.reshape(cur_index, (-1)), 1, -2)

        # Convert the boolean attention mask to an attention bias.
        if mask is not None:
            # attention mask in the form of attention bias
            attention_bias = lax.select(
                mask > 0, jnp.full(mask.shape, 0.0).astype(self.dtype), jnp.full(mask.shape, -1e10).astype(self.dtype)
            )
        else:
            attention_bias = None

        # Add provided bias term (e.g. relative position embedding).
        if bias is not None:
            attention_bias = combine_biases(attention_bias, bias)

        dropout_rng = None
        if not deterministic and self.dropout_rate > 0.0:
            dropout_rng = self.make_rng("dropout")

        # Apply attention.
        x = dot_product_attention(
            query,
            key,
            value,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
            dtype=self.dtype,
            float32_logits=self.float32_logits,
        )

        # Back to the original inputs dimensions.
        out = DenseGeneral(
            features=inputs_q.shape[-1],  # output dim is set to the input dim.
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            kernel_axes=("heads", "kv", "embed"),
            dtype=self.dtype,
            name="out",
        )(x)
        return out


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
    # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
    return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


def _canonicalize_tuple(x):
    if isinstance(x, Iterable):
        return tuple(x)
    else:
        return (x,)


# ------------------------------------------------------------------------------
# DenseGeneral for attention layers.
# ------------------------------------------------------------------------------
class DenseGeneral(nn.Module):
    """A linear transformation (without bias) with flexible axes.

    Attributes:
      features: tuple with numbers of output features.
      axis: tuple with axes to apply the transformation on.
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
    """

    features: Union[Iterable[int], int]
    axis: Union[Iterable[int], int] = -1
    dtype: DType = jnp.float32
    params_dtype: DType = jnp.float32
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal")
    kernel_axes: Tuple[str, ...] = ()
    use_bias: bool = True
    bias_init: Any = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along multiple dimensions.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        features = _canonicalize_tuple(self.features)
        axis = _canonicalize_tuple(self.axis)

        inputs = jnp.asarray(inputs, self.dtype)
        axis = _normalize_axes(axis, inputs.ndim)

        kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + features
        kernel_in_axis = np.arange(len(axis))
        kernel_out_axis = np.arange(len(axis), len(axis) + len(features))
        kernel = param_with_axes(
            "kernel",
            self.kernel_init,
            kernel_shape,
            self.params_dtype,
            kernel_in_axis,
            kernel_out_axis,
            axes=self.kernel_axes,
        )
        if self.use_bias:
            bias = param_with_axes("bias", self.bias_init, features, self.params_dtype, axes=(self.kernel_axes[-1],))
        kernel = jnp.asarray(kernel, self.dtype)

        contract_ind = tuple(range(0, len(axis)))
        y = lax.dot_general(inputs, kernel, ((axis, contract_ind), ((), ())))
        if self.use_bias:
            bias = jnp.asarray(bias, self.dtype)
            # y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
            y += jnp.reshape(bias, (1,) * (len(features) - y.ndim) + bias.shape[:])
        return y


def _convert_to_activation_function(fn_or_string: Union[str, Callable]) -> Callable:
    """Convert a string to an activation function."""
    if fn_or_string == "linear":
        return lambda x: x
    elif isinstance(fn_or_string, str):
        return getattr(nn, fn_or_string)
    elif callable(fn_or_string):
        return fn_or_string
    else:
        raise ValueError("don't know how to convert %s to an activation function" % (fn_or_string,))


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block.

    Attributes:
      intermediate_dim: Shared dimension of hidden layers.
      activations: Type of activations for each layer.  Each element is either
        'linear', a string function name in flax.linen, or a function.
      kernel_init: Kernel function, passed to the dense layers.
      deterministic: Whether the dropout layers should be deterministic.
      intermediate_dropout_rate: Dropout rate used after the intermediate layers.
      dtype: Type for the dense layer.
    """

    intermediate_dim: int = 2048
    activations: Sequence[Union[str, Callable]] = ("relu",)
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal")
    intermediate_dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
        """Applies Transformer MlpBlock module."""
        # Iterate over specified MLP input activation functions.
        # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
        activations = []
        for idx, act_fn in enumerate(self.activations):
            dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
            x = DenseGeneral(
                self.intermediate_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                kernel_axes=("embed", "mlp"),
                name=dense_name,
            )(inputs)
            x = _convert_to_activation_function(act_fn)(x)
            activations.append(x)

        # Take elementwise product of above intermediate activations.
        x = functools.reduce(operator.mul, activations)
        # Apply dropout and final dense output projection.
        x = nn.Dropout(rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic
        )  # Broadcast along length.
        x = with_sharding_constraint(x, ("batch", "length", "mlp"))
        output = DenseGeneral(
            inputs.shape[-1], dtype=self.dtype, kernel_init=self.kernel_init, kernel_axes=("mlp", "embed"), name="wo"
        )(x)
        return output


class Embed(nn.Module):
    """A parameterized function from integers [0, n) to d-dimensional vectors.

    Attributes:
      num_embeddings: number of embeddings.
      features: number of feature dimensions for each embedding.
      dtype: the dtype of the embedding vectors (default: float32).
      embedding_init: embedding initializer.
      one_hot: performs the gather with a one-hot contraction rather than a true
        gather. This is currently needed for SPMD partitioning.
    """

    num_embeddings: int
    features: int
    cast_input_dtype: Optional[DType] = None
    dtype: DType = jnp.float32
    params_dtype: DType = jnp.float32
    attend_dtype: Optional[DType] = None
    embedding_init: Initializer = default_embed_init
    one_hot: bool = True
    embedding: Array = dataclasses.field(init=False)

    def setup(self):
        self.embedding = param_with_axes(
            "embedding",
            self.embedding_init,
            (self.num_embeddings, self.features),
            self.params_dtype,
            axes=("vocab", "embed"),
        )

    def __call__(self, inputs: Array) -> Array:
        """Embeds the inputs along the last dimension.

        Args:
          inputs: input data, all dimensions are considered batch dimensions.

        Returns:
          Output which is embedded input data.  The output shape follows the input,
          with an additional `features` dimension appended.
        """
        if self.cast_input_dtype:
            inputs = inputs.astype(self.cast_input_dtype)
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Input type must be an integer or unsigned integer.")
        if self.one_hot:
            iota = lax.iota(jnp.int32, self.num_embeddings)
            one_hot = jnp.array(inputs[..., jnp.newaxis] == iota, dtype=self.dtype)
            output = jnp.dot(one_hot, jnp.asarray(self.embedding, self.dtype))
        else:
            output = jnp.asarray(self.embedding, self.dtype)[inputs]
            output = with_sharding_constraint(output, ("batch", "length", "embed"))
        return output

    def attend(self, query: Array) -> Array:
        """Attend over the embedding using a query array.

        Args:
          query: array with last dimension equal the feature depth `features` of the
            embedding.

        Returns:
          An array with final dim `num_embeddings` corresponding to the batched
          inner-product of the array of query vectors against each embedding.
          Commonly used for weight-sharing between embeddings and logit transform
          in NLP models.
        """
        dtype = self.attend_dtype if self.attend_dtype is not None else self.dtype
        return jnp.dot(query, jnp.asarray(self.embedding, dtype).T)


class RelativePositionBiases(nn.Module):
    """Adds T5-style relative positional embeddings to the attention logits.

    Attributes:
      num_buckets: Number of buckets to bucket distances between key and query
        positions into.
      max_distance: Maximum distance before everything is lumped into the last
        distance bucket.
      num_heads: Number of heads in the attention layer. Each head will get a
        different relative position weighting.
      dtype: Type of arrays through this module.
      embedding_init: initializer for relative embedding table.
    """

    num_buckets: int
    max_distance: int
    num_heads: int
    dtype: Any
    embedding_init: Callable[..., Array] = nn.linear.default_embed_init

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """Translate relative position to a bucket number for relative attention.

        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger
        buckets for larger absolute relative_positions.  All relative
        positions >=max_distance  map to the same bucket.  All relative
        positions <=-max_distance map to the same bucket.  This should allow for
        more graceful generalization to longer sequences than the model has been
        trained on.

        Args:
          relative_position: an int32 array
          bidirectional: a boolean - whether the attention is bidirectional
          num_buckets: an integer
          max_distance: an integer

        Returns:
          a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).astype(np.int32) * num_buckets
            n = np.abs(n)
        else:
            n = np.maximum(n, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            np.log(n.astype(np.float32) / max_exact + np.finfo(np.float32).eps)
            / np.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(np.int32)
        val_if_large = np.minimum(val_if_large, num_buckets - 1)
        ret += np.where(is_small, n, val_if_large)
        return ret

    @nn.compact
    def __call__(self, qlen, klen, bidirectional=True):
        """Produce relative position embedding attention biases.

        Args:
          qlen: attention query length.
          klen: attention key length.
          bidirectional: whether to allow positive memory-query relative position
            embeddings.

        Returns:
          output: `(1, len, q_len, k_len)` attention bias
        """
        # TODO(levskaya): should we be computing this w. numpy as a program
        # constant?
        context_position = np.arange(qlen, dtype=jnp.int32)[:, None]
        memory_position = np.arange(klen, dtype=jnp.int32)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        relative_attention_bias = param_with_axes(
            "rel_embedding",
            self.embedding_init,
            (self.num_heads, self.num_buckets),
            jnp.float32,
            axes=("heads", "relpos_buckets"),
        )

        relative_attention_bias = jnp.asarray(relative_attention_bias, self.dtype)
        # Instead of using a slow gather, we create a leading-dimension one-hot
        # array from rp_bucket and use it to perform the gather-equivalent via a
        # contraction, i.e.:
        # (num_head, num_buckets) x (num_buckets one-hot, qlen, klen).
        # This is equivalent to relative_attention_bias[:, rp_bucket]
        bcast_iota = lax.broadcasted_iota(jnp.int32, (self.num_buckets, 1, 1), 0)
        rp_bucket_one_hot = jnp.array(rp_bucket[jnp.newaxis, ...] == bcast_iota, dtype=self.dtype)
        # --> shape (qlen, klen, num_heads)
        values = lax.dot_general(
            relative_attention_bias, rp_bucket_one_hot, (((1,), (0,)), ((), ()))  # rhs, lhs contracting dims
        )  # no batched dims
        # Add a singleton batch dimension.
        # --> shape (1, num_heads, qlen, klen)
        return values[jnp.newaxis, ...]


# ------------------------------------------------------------------------------
# T5 Layernorm - no subtraction of mean or bias.
# ------------------------------------------------------------------------------
# class LayerNorm(nn.Module):
#   """T5 Layer normalization operating on the last axis of the input data."""
#   epsilon: float = 1e-6
#   dtype: Any = jnp.float32
#   scale_init: Initializer = nn.initializers.ones

#   @nn.compact
#   def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#     """Applies layer normalization on the input."""
#     x = jnp.asarray(x, jnp.float32)
#     features = x.shape[-1]
#     mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
#     y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
#     scale = param_with_axes(
#         'scale', self.scale_init, (features,), jnp.float32, axes=('embed',))

#     scale = jnp.asarray(scale, self.dtype)
#     return y * scale


class LayerNorm(nn.Module):
    """Layer normalization (https://arxiv.org/abs/1607.06450).
    Operates on the last axis of the input data.
    It normalizes the activations of the layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within
    each example close to 0 and the activation standard deviation close to 1.
    Attributes:
      epsilon: A small float added to variance to avoid dividing by zero.
      dtype: the dtype of the computation (default: float32).
      use_bias:  If True, bias (beta) is added.
      use_scale: If True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: Initializer for bias, by default, zero.
      scale_init: Initializer for scale, by default, one.
    """

    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    params_dtype: DType = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Any], Array] = nn.initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Any], Array] = nn.initializers.ones

    @nn.compact
    def __call__(self, x):
        """Applies layer normalization on the input.
        Args:
          x: the inputs
        Returns:
          Normalized inputs (the same shape as inputs).
        """
        x = jnp.asarray(x, jnp.float32)
        features = x.shape[-1]
        mean = jnp.mean(x, axis=-1, keepdims=True)
        mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
        var = mean2 - lax.square(mean)
        mul = lax.rsqrt(var + self.epsilon)
        if self.use_scale:
            scale = param_with_axes("scale", self.scale_init, (features,), self.params_dtype, axes=("embed",))
            mul = mul * jnp.asarray(scale, self.dtype)
        y = (x - mean) * mul
        if self.use_bias:
            bias = param_with_axes("bias", self.bias_init, (features,), self.params_dtype, axes=("embed",))
            y = y + jnp.asarray(bias, self.dtype)
        return jnp.asarray(y, self.dtype)


# ------------------------------------------------------------------------------
# Mask-making utility functions.
# ------------------------------------------------------------------------------
def make_attention_mask(
    query_input: Array,
    key_input: Array,
    pairwise_fn: Callable = jnp.multiply,
    extra_batch_dims: int = 0,
    dtype: DType = jnp.float32,
) -> Array:
    """Mask-making helper for attention weights.

    In case of 1d inputs (i.e., `[batch, len_q]`, `[batch, len_kv]`, the
    attention weights will be `[batch, heads, len_q, len_kv]` and this
    function will produce `[batch, 1, len_q, len_kv]`.

    Args:
      query_input: a batched, flat input of query_length size
      key_input: a batched, flat input of key_length size
      pairwise_fn: broadcasting elementwise comparison function
      extra_batch_dims: number of extra batch dims to add singleton axes for, none
        by default
      dtype: mask return dtype

    Returns:
      A `[batch, 1, len_q, len_kv]` shaped mask for 1d attention.
    """
    # [batch, len_q, len_kv]
    mask = pairwise_fn(
        # [batch, len_q] -> [batch, len_q, 1]
        jnp.expand_dims(query_input, axis=-1),
        # [batch, len_q] -> [batch, 1, len_kv]
        jnp.expand_dims(key_input, axis=-2),
    )

    # [batch, 1, len_q, len_kv]. This creates the head dim.
    mask = jnp.expand_dims(mask, axis=-3)
    mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
    return mask.astype(dtype)


def make_causal_mask(x: Array, extra_batch_dims: int = 0, dtype: DType = jnp.float32) -> Array:
    """Make a causal mask for self-attention.

    In case of 1d inputs (i.e., `[batch, len]`, the self-attention weights
    will be `[batch, heads, len, len]` and this function will produce a
    causal mask of shape `[batch, 1, len, len]`.

    Note that a causal mask does not depend on the values of x; it only depends on
    the shape. If x has padding elements, they will not be treated in a special
    manner.

    Args:
      x: input array of shape `[batch, len]`
      extra_batch_dims: number of batch dims to add singleton axes for, none by
        default
      dtype: mask return dtype

    Returns:
      A `[batch, 1, len, len]` shaped causal mask for 1d attention.
    """
    idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
    return make_attention_mask(idxs, idxs, jnp.greater_equal, extra_batch_dims=extra_batch_dims, dtype=dtype)


def combine_masks(*masks: Optional[Array], dtype: DType = jnp.float32):
    """Combine attention masks.

    Args:
      *masks: set of attention mask arguments to combine, some can be None.
      dtype: final mask dtype

    Returns:
      Combined mask, reduced by logical and, returns None if no masks given.
    """
    masks = [m for m in masks if m is not None]
    if not masks:
        return None
    assert all(
        (x.ndim == masks[0].ndim for x in masks)
    ), f"masks must have same rank: {tuple((x.ndim for x in masks))}"
    mask, *other_masks = masks
    for other_mask in other_masks:
        mask = jnp.logical_and(mask, other_mask)
    return mask.astype(dtype)


def combine_biases(*masks: Optional[Array]):
    """Combine attention biases.

    Args:
      *masks: set of attention bias arguments to combine, some can be None.

    Returns:
      Combined mask, reduced by summation, returns None if no masks given.
    """
    masks = [m for m in masks if m is not None]
    if not masks:
        return None
    assert all(
        (x.ndim == masks[0].ndim for x in masks)
    ), f"masks must have same rank: {tuple((x.ndim for x in masks))}"
    mask, *other_masks = masks
    for other_mask in other_masks:
        mask = mask + other_mask
    return mask


def make_decoder_mask(
    decoder_target_tokens: Array,
    dtype: DType,
    decoder_causal_attention: Optional[Array] = None,
    decoder_segment_ids: Optional[Array] = None,
) -> Array:
    """Compute the self-attention mask for a decoder.

    Decoder mask is formed by combining a causal mask, a padding mask and an
    optional packing mask. If decoder_causal_attention is passed, it makes the
    masking non-causal for positions that have value of 1.

    A prefix LM is applied to a dataset which has a notion of "inputs" and
    "targets", e.g., a machine translation task. The inputs and targets are
    concatenated to form a new target. `decoder_target_tokens` is the concatenated
    decoder output tokens.

    The "inputs" portion of the concatenated sequence can attend to other "inputs"
    tokens even for those at a later time steps. In order to control this
    behavior, `decoder_causal_attention` is necessary. This is a binary mask with
    a value of 1 indicating that the position belonged to "inputs" portion of the
    original dataset.

    Example:

      Suppose we have a dataset with two examples.

      ds = [{"inputs": [6, 7], "targets": [8]},
            {"inputs": [3, 4], "targets": [5]}]

      After the data preprocessing with packing, the two examples are packed into
      one example with the following three fields (some fields are skipped for
      simplicity).

         decoder_target_tokens = [[6, 7, 8, 3, 4, 5, 0]]
           decoder_segment_ids = [[1, 1, 1, 2, 2, 2, 0]]
      decoder_causal_attention = [[1, 1, 0, 1, 1, 0, 0]]

      where each array has [batch, length] shape with batch size being 1. Then,
      this function computes the following mask.

                        mask = [[[[1, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 1, 0, 0],
                                  [0, 0, 0, 1, 1, 0, 0],
                                  [0, 0, 0, 1, 1, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 0]]]]

      mask[b, 1, :, :] represents the mask for the example `b` in the batch.
      Because mask is for a self-attention layer, the mask's shape is a square of
      shape [query length, key length].

      mask[b, 1, i, j] = 1 means that the query token at position i can attend to
      the key token at position j.

    Args:
      decoder_target_tokens: decoder output tokens. [batch, length]
      dtype: dtype of the output mask.
      decoder_causal_attention: a binary mask indicating which position should
        only attend to earlier positions in the sequence. Others will attend
        bidirectionally. [batch, length]
      decoder_segment_ids: decoder segmentation info for packed examples. [batch,
        length]

    Returns:
      the combined decoder mask.
    """
    masks = []
    # The same mask is applied to all attention heads. So the head dimension is 1,
    # i.e., the mask will be broadcast along the heads dim.
    # [batch, 1, length, length]
    causal_mask = make_causal_mask(decoder_target_tokens, dtype=dtype)

    # Positions with value 1 in `decoder_causal_attneition` can attend
    # bidirectionally.
    if decoder_causal_attention is not None:
        # [batch, 1, length, length]
        inputs_mask = make_attention_mask(
            decoder_causal_attention, decoder_causal_attention, jnp.logical_and, dtype=dtype
        )
        masks.append(jnp.logical_or(causal_mask, inputs_mask).astype(dtype))
    else:
        masks.append(causal_mask)

    # Padding mask.
    masks.append(make_attention_mask(decoder_target_tokens > 0, decoder_target_tokens > 0, dtype=dtype))

    # Packing mask
    if decoder_segment_ids is not None:
        masks.append(make_attention_mask(decoder_segment_ids, decoder_segment_ids, jnp.equal, dtype=dtype))

    return combine_masks(*masks, dtype=dtype)


def canonicalize_padding(padding: PaddingLike, rank: int) -> LaxPadding:
    """ "Canonicalizes conv padding to a jax.lax supported format."""
    if isinstance(padding, str):
        return padding
    if isinstance(padding, int):
        return [(padding, padding)] * rank
    if isinstance(padding, Sequence) and len(padding) == rank:
        new_pad = []
        for p in padding:
            if isinstance(p, int):
                new_pad.append((p, p))
            elif isinstance(p, tuple) and len(p) == 2:
                new_pad.append(p)
            else:
                break
        if len(new_pad) == rank:
            return new_pad
    raise ValueError(
        f"Invalid padding format: {padding}, should be str, int,"
        f" or a sequence of len {rank} where each element is an"
        f" int or pair of ints."
    )


def _conv_dimension_numbers(input_shape):
    """Computes the dimension numbers based on the input shape."""
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


class _Conv(nn.Module):
    """Convolution Module wrapping `lax.conv_general_dilated[_local]`.

    Attributes:
      features: number of convolution filters.
      kernel_size: shape of the convolutional kernel. For 1D convolution,
        the kernel size can be passed as an integer. For all other cases, it must
        be a sequence of integers.
      strides: an integer or a sequence of `n` integers, representing the
        inter-window strides (default: 1).
      padding: either the string `'SAME'`, the string `'VALID'`, the string
        `'CIRCULAR'` (periodic boundary conditions), or a sequence of `n` `(low,
        high)` integer pairs that give the padding to apply before and after each
        spatial dimension. A single int is interpeted as applying the same padding
        in all dims and passign a single int in a sequence causes the same padding
        to be used on both sides. `'CAUSAL'` padding for a 1D convolution will
        left-pad the convolution axis, resulting in same-sized output.
      input_dilation: an integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of `inputs`
        (default: 1). Convolution with input dilation `d` is equivalent to
        transposed convolution with stride `d`.
      kernel_dilation: an integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel (default: 1). Convolution with kernel dilation
        is also known as 'atrous convolution'.
      feature_group_count: integer, default 1. If specified divides the input
        features into groups.
      use_bias: whether to add a bias to the output (default: True).
      mask: Optional mask for the weights during masked convolution. The mask must
            be the same shape as the convolution weight matrix.
      dtype: the dtype of the computation (default: infer from input and params).
      params_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the convolutional kernel.
      bias_init: initializer for the bias.
    """

    features: int
    kernel_size: Sequence[int]
    strides: Union[None, int, Sequence[int]] = 1
    padding: PaddingLike = "SAME"
    input_dilation: Union[None, int, Sequence[int]] = 1
    kernel_dilation: Union[None, int, Sequence[int]] = 1
    feature_group_count: int = 1
    use_bias: bool = True
    mask: Optional[Array] = None
    dtype: Optional[DType] = None
    params_dtype: DType = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, DType], Array] = nn.initializers.lecun_normal()
    bias_init: Callable[[PRNGKey, Shape, DType], Array] = nn.initializers.zeros
    conv_general_dilated: ConvGeneralDilatedT = lax.conv_general_dilated
    kernel_axes: Tuple[str, ...] = ()

    @property
    def shared_weights(self) -> bool:  # type: ignore
        """Defines whether weights are shared or not between different pixels.

        Returns:
          `True` to use shared weights in convolution (regular convolution).
          `False` to use different weights at different pixels, a.k.a.
          "locally connected layer", "unshared convolution", or "local convolution".

        """
        ...

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a (potentially unshared) convolution to the inputs.

        Args:
          inputs: input data with dimensions (*batch_dims, spatial_dims...,
            features). This is the channels-last convention, i.e. NHWC for a 2d
            convolution and NDHWC for a 3D convolution. Note: this is different from
            the input convention used by `lax.conv_general_dilated`, which puts the
            spatial dimensions last.
            Note: If the input has more than 1 batch dimension, all batch dimensions
            are flattened into a single dimension for the convolution and restored
            before returning.  In some cases directly vmap'ing the layer may yield
            better performance than this default flattening approach.  If the input
            lacks a batch dimension it will be added for the convolution and removed
            n return, an allowance made to enable writing single-example code.

        Returns:
          The convolved data.
        """

        if isinstance(self.kernel_size, int):
            raise TypeError(
                "Expected Conv kernel_size to be a"
                " tuple/list of integers (eg.: [3, 3]) but got"
                f" {self.kernel_size}."
            )
        else:
            kernel_size = tuple(self.kernel_size)

        def maybe_broadcast(x: Optional[Union[int, Sequence[int]]]) -> Tuple[int, ...]:
            if x is None:
                # backward compatibility with using None as sentinel for
                # broadcast 1
                x = 1
            if isinstance(x, int):
                return (x,) * len(kernel_size)
            return tuple(x)

        # Combine all input batch dimensions into a single leading batch axis.
        num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
        if num_batch_dimensions != 1:
            input_batch_shape = inputs.shape[:num_batch_dimensions]
            total_batch_size = int(np.prod(input_batch_shape))
            flat_input_shape = (total_batch_size,) + inputs.shape[num_batch_dimensions:]
            inputs = jnp.reshape(inputs, flat_input_shape)

        # self.strides or (1,) * (inputs.ndim - 2)
        strides = maybe_broadcast(self.strides)
        input_dilation = maybe_broadcast(self.input_dilation)
        kernel_dilation = maybe_broadcast(self.kernel_dilation)

        padding_lax = canonicalize_padding(self.padding, len(kernel_size))
        if padding_lax == "CIRCULAR":
            kernel_size_dilated = [(k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)]
            zero_pad: List[Tuple[int, int]] = [(0, 0)]
            pads = zero_pad + [((k - 1) // 2, k // 2) for k in kernel_size_dilated] + [(0, 0)]
            inputs = jnp.pad(inputs, pads, mode="wrap")
            padding_lax = "VALID"
        elif padding_lax == "CAUSAL":
            if len(kernel_size) != 1:
                raise ValueError("Causal padding is only implemented for 1D convolutions.")
            left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
            pads = [(0, 0), (left_pad, 0), (0, 0)]
            inputs = jnp.pad(inputs, pads)
            padding_lax = "VALID"

        dimension_numbers = _conv_dimension_numbers(inputs.shape)
        in_features = jnp.shape(inputs)[-1]

        if self.shared_weights:
            # One shared convolutional kernel for all pixels in the output.
            assert in_features % self.feature_group_count == 0
            kernel_shape = kernel_size + (in_features // self.feature_group_count, self.features)

        else:
            if self.feature_group_count != 1:
                raise NotImplementedError(
                    f"`lax.conv_general_dilated_local` does not support "
                    f"`feature_group_count != 1`, got `{self.feature_group_count}`."
                )

            # Need to know the spatial output shape of a standard convolution to
            # create the unshared convolution kernel.
            conv_output_shape = jax.eval_shape(
                lambda lhs, rhs: self.conv_general_dilated(  # pylint: disable=g-long-lambda
                    lhs=lhs,
                    rhs=rhs,
                    window_strides=strides,
                    padding=padding_lax,
                    dimension_numbers=dimension_numbers,
                    lhs_dilation=input_dilation,
                    rhs_dilation=kernel_dilation,
                ),
                inputs,
                jax.ShapedArray(kernel_size + (in_features, self.features), inputs.dtype),
            ).shape

            # One (unshared) convolutional kernel per each pixel in the output.
            kernel_shape = conv_output_shape[1:-1] + (np.prod(kernel_size) * in_features, self.features)

        if self.mask is not None and self.mask.shape != kernel_shape:
            raise ValueError(
                "Mask needs to have the same shape as weights. " f"Shapes are: {self.mask.shape}, {kernel_shape}"
            )

        kernel = param_with_axes(
            "kernel",
            self.kernel_init,
            kernel_shape,
            self.params_dtype,
            axes=self.kernel_axes,
        )

        if self.mask is not None:
            kernel *= self.mask

        if self.use_bias:
            if self.shared_weights:
                # One bias weight per output channel, shared between pixels.
                bias_shape = (self.features,)
            else:
                # One bias weight per output entry, unshared betwen pixels.
                bias_shape = conv_output_shape[1:]

            bias = param_with_axes(
                "bias",
                self.bias_init,
                bias_shape,
                self.params_dtype,
                axes=(self.kernel_axes[-1],),
            )
        else:
            bias = None

        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
        if self.shared_weights:
            y = self.conv_general_dilated(
                inputs,
                kernel,
                strides,
                padding_lax,
                lhs_dilation=input_dilation,
                rhs_dilation=kernel_dilation,
                dimension_numbers=dimension_numbers,
                feature_group_count=self.feature_group_count,
                precision=self.precision,
            )
        else:
            y = lax.conv_general_dilated_local(
                lhs=inputs,
                rhs=kernel,
                window_strides=strides,
                padding=padding_lax,
                filter_shape=kernel_size,
                lhs_dilation=input_dilation,
                rhs_dilation=kernel_dilation,
                dimension_numbers=dimension_numbers,
                precision=self.precision,
            )

        if self.use_bias:
            bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
            y += bias

        if num_batch_dimensions != 1:
            output_shape = input_batch_shape + y.shape[1:]
            y = jnp.reshape(y, output_shape)
        return y


class Conv(_Conv):
    """Convolution Module wrapping `lax.conv_general_dilated`.

    Attributes:
      features: number of convolution filters.
      kernel_size: shape of the convolutional kernel. For 1D convolution,
        the kernel size can be passed as an integer. For all other cases, it must
        be a sequence of integers.
      strides: an integer or a sequence of `n` integers, representing the
        inter-window strides (default: 1).
      padding: either the string `'SAME'`, the string `'VALID'`, the string
        `'CIRCULAR'` (periodic boundary conditions), or a sequence of `n` `(low,
        high)` integer pairs that give the padding to apply before and after each
        spatial dimension. A single int is interpeted as applying the same padding
        in all dims and passign a single int in a sequence causes the same padding
        to be used on both sides. `'CAUSAL'` padding for a 1D convolution will
        left-pad the convolution axis, resulting in same-sized output.
      input_dilation: an integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of `inputs`
        (default: 1). Convolution with input dilation `d` is equivalent to
        transposed convolution with stride `d`.
      kernel_dilation: an integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel (default: 1). Convolution with kernel dilation
        is also known as 'atrous convolution'.
      feature_group_count: integer, default 1. If specified divides the input
        features into groups.
      use_bias: whether to add a bias to the output (default: True).
      mask: Optional mask for the weights during masked convolution. The mask must
            be the same shape as the convolution weight matrix.
      dtype: the dtype of the computation (default: infer from input and params).
      params_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the convolutional kernel.
      bias_init: initializer for the bias.
    """

    @property
    def shared_weights(self) -> bool:
        return True
