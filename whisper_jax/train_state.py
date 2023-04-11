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

"""Utilities for partitioning."""

from typing import Any, Mapping, MutableMapping, Optional, Tuple

import flax.core
import flax.serialization
import flax.struct
import jax.numpy as jnp
from flax import traverse_util
from flax.core import scope as flax_scope
from flax.linen import partitioning as flax_partitioning


EMPTY_DICT = flax.core.freeze({})
FrozenDict = flax_scope.FrozenDict
FrozenVariableDict = flax_scope.FrozenVariableDict
MutableVariableDict = flax_scope.MutableVariableDict
VariableDict = flax_scope.VariableDict


def _validate_params_axes(params_axes, params):
    axis_names = flax_partitioning.get_axis_names(params_axes)
    missing_params_axes = set(traverse_util.flatten_dict(params, sep="/")) - set(
        traverse_util.flatten_dict(axis_names, sep="/")
    )
    if missing_params_axes:
        raise ValueError(f"Missing axis names for parameters: {missing_params_axes}")


def _split_variables_and_axes(variables_and_axes: FrozenVariableDict) -> Tuple[FrozenVariableDict, FrozenVariableDict]:
    """Splits `variables_and_axes` into two separate dicts with the same keys."""
    # For each `key`, `key_axes` (if any) are its axes in `variables_and_axes`.
    variables = {}
    axes = {}
    for k, v in variables_and_axes.items():
        if k.endswith("_axes"):
            axes[k[:-5]] = v  # k without "_axes".
            _validate_params_axes(v, variables_and_axes[k[:-5]])  # k without "_axes".
        else:
            variables[k] = v
    return flax.core.freeze(variables), flax.core.freeze(axes)


class InferenceState(flax.struct.PyTreeNode):
    """State compatible with FlaxOptimTrainState without optimizer state."""

    step: jnp.ndarray
    params: flax_scope.FrozenVariableDict
    params_axes: Optional[flax_scope.FrozenVariableDict] = None
    flax_mutables: flax_scope.FrozenDict = EMPTY_DICT
    flax_mutables_axes: Optional[flax_scope.FrozenVariableDict] = None

    @classmethod
    def create(cls, model_variables: FrozenVariableDict) -> "InferenceState":
        other_variables, params = model_variables.pop("params")
        if "params_axes" in other_variables:
            other_variables, params_axes = other_variables.pop("params_axes")
            _validate_params_axes(params_axes, params)
        else:
            params_axes = None

        # Split other_variables into mutables and their corresponding axes.
        flax_mutables, flax_mutables_axes = _split_variables_and_axes(other_variables)
        flax_mutables_axes = flax_mutables_axes or None
        return InferenceState(
            step=jnp.array(0),
            params=params,
            params_axes=params_axes,
            flax_mutables=flax_mutables,
            flax_mutables_axes=flax_mutables_axes,
        )

    @property
    def param_states(self) -> FrozenVariableDict:
        """The optimizer states of the parameters as a PyTree."""
        raise NotImplementedError("InferenceState has no optimizer states.")

    def apply_gradient(self, *args, **kwargs) -> "InferenceState":
        raise NotImplementedError("InferenceState does not support `apply_gradient`.")

    def state_dict(self) -> MutableMapping[str, Any]:
        state_dict = {"target": flax.core.unfreeze(self.params), "state": {"step": self.step}}
        if self.flax_mutables:
            state_dict["flax_mutables"] = flax.core.unfreeze(self.flax_mutables)
        return state_dict

    def replace_step(self, step: jnp.ndarray) -> "InferenceState":
        return self.replace(step=step)

    def replace_params(self, params: FrozenVariableDict) -> "InferenceState":
        return self.replace(params=params)

    def replace_flax_mutables(self, flax_mutables: FrozenDict) -> "InferenceState":
        return self.replace(flax_mutables=flax_mutables)

    def restore_state(self, state_dict: Mapping[str, Any]) -> "InferenceState":
        return self.replace(
            params=flax.core.freeze(state_dict["target"]),
            step=state_dict["state"]["step"],
            flax_mutables=flax.core.freeze(state_dict["flax_mutables"])
            if "flax_mutables" in state_dict
            else EMPTY_DICT,
        )

    def as_logical_axes(self) -> "InferenceState":
        # Set step to None so that when the logical axes are processed by the
        # flax.partitioning.logical_to_mesh_axes function, it will be skipped
        # because jax.tree_map will short circut and never call the function on the
        # step.
        flax_mutables_axes = self.flax_mutables_axes or EMPTY_DICT
        return InferenceState(
            step=None,
            params=flax_partitioning.get_axis_names(self.params_axes),
            flax_mutables=flax_partitioning.get_axis_names(flax_mutables_axes),
        )
