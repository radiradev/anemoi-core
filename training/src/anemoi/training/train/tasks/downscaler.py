# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.training.losses.scalers.base_scaler import AvailableCallbacks

from .forecaster import GraphForecaster

if TYPE_CHECKING:
    from collections.abc import Generator

    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.schemas.base_schema import BaseSchema

LOGGER = logging.getLogger(__name__)


class GraphDiffusionDownscaler(BaseGraphModule):
    """Graph neural network downscaler for diffusion."""

    def __init__(
        self,
        *,
        config: BaseSchema,
        graph_data: HeteroData,
        truncation_data: dict,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:

        super().__init__(
            config=config,
            graph_data=graph_data,
            truncation_data=truncation_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        self.rho = config.model.model.diffusion.rho

    def forward(
        self, x: torch.Tensor, y_noised: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        return self.model.model.fwd_with_preconditioning(
            x,
            y_noised,
            sigma,
            model_comm_group=self.model_comm_group,
            grid_shard_shapes=self.grid_shard_shapes,
        )

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        weights: torch.Tensor,
        grid_shard_slice: slice | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Compute the diffusion loss with noise weighting.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values
        y : torch.Tensor
            Target values
        grid_shard_slice : slice | None
            Grid shard slice for distributed training
        weights : torch.Tensor
            Noise weights for diffusion loss computation
        **_kwargs
            Additional arguments

        Returns
        -------
        torch.Tensor
            Computed loss with noise weighting applied
        """
        return self.loss(
            y_pred,
            y,
            weights=weights,
            grid_shard_slice=grid_shard_slice,
            group=self.model_comm_group,
        )

    def _step(
        self,
        batch: list[torch.Tensor],
        batch_idx: int,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        del batch_idx
        training_mode = True
        loss = torch.zeros(
            1, dtype=batch.dtype, device=self.device, requires_grad=False
        )
        metrics = {}
        y_preds = []

        """ ROllout step """
        x_in, x_in_hres, Y = batch
        # Y = Y[:, :, :, ..., self.data_indices.data.output.full] #(see if necessary)

        # Residuals prediction
        x_in_interp_to_hres = self.model.interpolate_down(
            x_in[:, 0, 0, ...], grad_checkpoint=False
        )[:, None, None, ...]
        self.x_in_matching_channel_indices = self.x_in_matching_channel_indices.to(
            x_in_interp_to_hres.device
        )
        y = y - x_in_interp_to_hres[..., self.x_in_matching_channel_indices]

        # Normalisation
        x_in_interp_to_hres = self.model.pre_processors(
            x_in_interp_to_hres, "input_lres"
        )  # need in place ?, in_place=False)
        # x_in_interp_to_hres = x_in_interp_to_hres[  :, :, ..., self.data_indices.data.input[0].full] (see if necessary)
        x_in_hres = self.model.pre_processors(
            x_in_hres, "input_hres"
        )  # , in_place=False
        # x_in_hres = x_in_hres[:, :, ..., self.data_indices.data.input[1].full]
        y = self.model.pre_processors(Y, "output")

        # Scaler update
        self.update_scalers(callback=AvailableCallbacks.ON_BATCH_START)

        # get noise level and associated loss weights
        sigma, noise_weights = self._get_noise_level(
            shape=(Y.shape[0],) + (1,) * (Y.ndim - 2),
            sigma_max=self.model.model.sigma_max,
            sigma_min=self.model.model.sigma_min,
            sigma_data=self.model.model.sigma_data,
            rho=self.rho,
            device=Y.device,
        )

        # get targets and noised targets
        y_noised = self._noise_target(y, sigma)

        # prediction, fwd_with_preconditioning
        y_pred = self(
            torch.cat((x_in_interp_to_hres, x_in_hres), dim=-1),
            y_noised,
            sigma,
        )  # shape is (bs, ens, latlon, nvar)

        # Use checkpoint for compute_loss_metrics
        loss, metrics_next = checkpoint(
            self.compute_loss_metrics,
            y_pred,
            y,
            training_mode,
            validation_mode,
            weights=noise_weights,
            use_reentrant=False,
        )

        return loss, metrics, y_preds

    def _noise_target(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Add noise to the state."""
        return x + torch.randn_like(x) * sigma

    def _get_noise_level(
        self,
        shape: torch.shape,
        sigma_max: float,
        sigma_min: float,
        sigma_data: float,
        rho: float,
        device: torch.device,
    ) -> tuple[torch.Tensor]:
        if self.training_approach == "probabilistic_high_noise":
            rnd_uniform = torch.rand(shape, device=device)
            sigma = (
                sigma_max ** (1.0 / rho)
                + rnd_uniform * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
            ) ** rho

        elif self.training_approach == "probabilistic":
            log_sigma = torch.normal(
                mean=self.lognormal_mean,
                std=self.lognormal_std,
                size=shape,
                device=device,
            )
            sigma = torch.exp(log_sigma)
        elif self.training_approach == "deterministic":
            sigma = torch.full(
                shape,
                fill_value=5000.0,
                device=device,
            )

        weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
        return sigma, weight


def match_tensor_channels(input_name_to_index, output_name_to_index):
    """
    Reorders and selects channels from input tensor to match output tensor structure.
    x_in: Input tensor of shape [batch, n_grid_points, channels]
    """

    common_channels = set(input_name_to_index.keys()) & set(output_name_to_index.keys())

    # for each output channel, look for corresponding input channel
    channel_mapping = []
    for channel_name in output_name_to_index.keys():
        if channel_name in common_channels:
            input_pos = input_name_to_index[channel_name]
            channel_mapping.append(input_pos)

    # Convert to tensor for indexing
    channel_indices = torch.tensor(channel_mapping)

    return channel_indices
