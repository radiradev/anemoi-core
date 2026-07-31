# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import einops
from torch import Tensor

if TYPE_CHECKING:
    from anemoi.models.models.base import BaseGraphModel


class DecodingTargetFeature(ABC):
    """Base class for a decoder target feature.
    
    Attributes
    ----------
    needs_sharding : bool
        Whether this feature must be explicitly sharded:
        - `True` for features derived from node attributes
        - `False` for features already sharded via the input tensor or the encoder
    """
    needs_sharding: bool = False

    @abstractmethod
    def dim(self, model: BaseGraphModel, dataset_name: str) -> int:
        """Return the feature dimension contributed by this target feature."""

    @abstractmethod
    def tensor(
        self,
        model: BaseGraphModel,
        x_input_data: Tensor,
        x_encoded_data: Tensor | None,
        batch_size: int,
        dataset_name: str,
    ) -> Tensor:
        """Extract the feature tensor of shape (batch*ensemble*grid, feature_dim)."""


class CoordinatesFeature(DecodingTargetFeature):
    """Sin/cos encoded lat-lon coordinates."""

    needs_sharding = True

    def dim(self, model: BaseGraphModel, dataset_name: str) -> int:
        return getattr(model.node_attributes, f"latlons_{dataset_name}").shape[1]

    def tensor(
        self,
        model: BaseGraphModel,
        x_input_data: Tensor,
        x_encoded_data: Tensor | None,
        batch_size: int,
        dataset_name: str,
    ) -> Tensor:
        coords = getattr(model.node_attributes, f"latlons_{dataset_name}")
        return einops.repeat(coords, "e f -> (repeat e) f", repeat=batch_size)


class InputForcingsFeature(DecodingTargetFeature):
    """Forcing variables over the output timestep window."""

    needs_sharding = False

    def dim(self, model: BaseGraphModel, dataset_name: str) -> int:
        return model.n_step_output * model.num_input_channels_forcings[dataset_name]

    def tensor(
        self,
        model: BaseGraphModel,
        x_input_data: Tensor,
        x_encoded_data: Tensor | None,
        batch_size: int,
        dataset_name: str,
    ) -> Tensor:
        indices = model._forcing_input_idx[dataset_name]
        x_forcing = x_input_data[:, : model.n_step_output, ..., indices]
        return einops.rearrange(x_forcing, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)")


class PrognosticsFeature(DecodingTargetFeature):
    """Prognostic variables over the input timestep window."""

    needs_sharding = False

    def dim(self, model: BaseGraphModel, dataset_name: str) -> int:
        return model.n_step_input * model.num_input_channels_prognostic[dataset_name]

    def tensor(
        self,
        model: BaseGraphModel,
        x_input_data: Tensor,
        x_encoded_data: Tensor | None,
        batch_size: int,
        dataset_name: str,
    ) -> Tensor:
        indices = model._internal_input_idx[dataset_name]
        x_prog = x_input_data[:, : model.n_step_input, ..., indices]
        return einops.rearrange(x_prog, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)")


class TrainableParametersFeature(DecodingTargetFeature):
    """Learnable per-node parameters."""

    needs_sharding = True

    def dim(self, model: BaseGraphModel, dataset_name: str) -> int:
        return model.node_attributes.num_trainable_parameters[dataset_name]

    def tensor(
        self,
        model: BaseGraphModel,
        x_input_data: Tensor,
        x_encoded_data: Tensor | None,
        batch_size: int,
        dataset_name: str,
    ) -> Tensor:
        trainable = model.node_attributes.trainable_tensors[dataset_name].trainable
        if trainable is None:
            decoder_name = model.dataset2decoder[dataset_name]
            raise ValueError(
                f"No trainable parameters configured for dataset '{dataset_name}'. "
                f"Set trainable_parameters.data > 0 or remove 'trainable_parameters' from "
                f"decoder.{decoder_name}.input_target_features."
            )
        return einops.repeat(trainable, "e f -> (repeat e) f", repeat=batch_size)


class EncodedDataFeature(DecodingTargetFeature):
    """Encoder-updated data tensor (full encoder output on data nodes)."""

    needs_sharding = False

    def dim(self, model: BaseGraphModel, dataset_name: str) -> int:
        return model._calculate_input_dim(dataset_name)

    def tensor(
        self,
        model: BaseGraphModel,
        x_input_data: Tensor,
        x_encoded_data: Tensor | None,
        batch_size: int,
        dataset_name: str,
    ) -> Tensor:
        if x_encoded_data is None:
            decoder_name = model.dataset2decoder[dataset_name]
            raise ValueError(
                f'"encoded_data" requires dataset "{dataset_name}" to have an encoder. '
                f"Update decoder.{decoder_name}.input_target_features."
            )
        return x_encoded_data


TARGET_FEATURE_REGISTRY: dict[str, DecodingTargetFeature] = {
    "coordinates": CoordinatesFeature(),
    "forcings": InputForcingsFeature(),
    "prognostics": PrognosticsFeature(),
    "trainable_parameters": TrainableParametersFeature(),
    "encoded_data": EncodedDataFeature(),
}

VALID_TARGET_FEATURES: set[str] = set(TARGET_FEATURE_REGISTRY.keys())
