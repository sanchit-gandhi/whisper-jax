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
    if params_axes is None:
        return  # No validation needed if params_axes is None
    axis_names = flax_partitioning.get_axis_names(params_axes)
    missing_params_axes = set(traverse_util.flatten_dict(params, sep="/")) - set(
        traverse_util.flatten_dict(axis_names, sep="/")
    )
    if missing_params_axes:
        raise ValueError(f"Missing axis names for parameters: {missing_params_axes}")


def _split_variables_and_axes(variables_and_axes: FrozenVariableDict) -> Tuple[FrozenVariableDict, FrozenVariableDict]:
    """Splits `variables_and_axes` into two separate dicts with the same keys."""
    variables = {}
    axes = {}
    for k, v in variables_and_axes.items():
        if k.endswith("_axes"):
            param_key = k[:-5]  # Remove "_axes" suffix
            axes[param_key] = v
            if param_key in variables_and_axes:
                _validate_params_axes(v, variables_and_axes[param_key])
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
        if "params" not in model_variables:
            raise ValueError("Missing 'params' in model_variables.")
        
        other_variables, params = model_variables.pop("params")
        params_axes = other_variables.pop("params_axes", None)
        _validate_params_axes(params_axes, params)
        
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

    def state_dict(self) -> MutableMapping[str, Any]:
        state_dict = {"target": flax.core.unfreeze(self.params), "state": {"step": self.step}}
        if self.flax_mutables:
            state_dict["flax_mutables"] = flax.core.unfreeze(self.flax_mutables)
        return state_dict

    def restore_state(self, state_dict: Mapping[str, Any]) -> "InferenceState":
        return self.replace(
            params=flax.core.freeze(state_dict["target"]),
            step=state_dict["state"]["step"],
            flax_mutables=flax.core.freeze(state_dict.get("flax_mutables", EMPTY_DICT)),
        )

    def as_logical_axes(self) -> "InferenceState":
        """Creates a logical axes representation of the inference state."""
        flax_mutables_axes = self.flax_mutables_axes or EMPTY_DICT
        return InferenceState(
            step=None,
            params=flax_partitioning.get_axis_names(self.params_axes) if self.params_axes else EMPTY_DICT,
            flax_mutables=flax_partitioning.get_axis_names(flax_mutables_axes),
        )
