# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional

import einops
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import get_shape_shards
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.training.data.data_handlers import SampleProvider
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiMultiModel(nn.Module):
    """Message passing graph neural network."""

    def __init__(
        self,
        *,
        sample_provider: SampleProvider,
        model_config: DotDict,
        graph_data: HeteroData,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        model_config : DotDict
            Model configuration
        graph_data : HeteroData
            Graph definition
        """
        super().__init__()
        model_config = DotDict(model_config)
        self._graph_data = graph_data
        self._graph_name_data = model_config.graph.data
        self._graph_name_hidden = model_config.graph.hidden

        self.multi_step = model_config.training.multistep_input
        self.num_channels = model_config.model.num_channels

        self.node_attributes = NamedNodesAttributes(model_config.model.trainable_parameters.hidden, self._graph_data)

        self.latent_residual_connection = True
        self.use_residual_connection = []  # ["era5"]

        sample_spec = sample_provider.spec
        self.input_channels: dict[str, int] = sample_spec.num_channels["input"]
        self.target_channels: dict[str, int] = sample_spec.num_channels["target"]
        self.input_names: list[str] = sample_spec.input_names
        self.target_names: list[str] = sample_spec.target_names

        self.node_attributes = NamedNodesAttributes(model_config.model.trainable_parameters.hidden, self._graph_data)

        # Encoder data -> hidden
        self.encoders = nn.ModuleDict({})
        for input_name in self.input_names:
            self.encoders[input_name] = instantiate(
                model_config.model.encoder,
                _recursive_=False,
                in_channels_src=self.node_attributes.attr_ndims[input_name] + self.input_channels[input_name],
                in_channels_dst=self.node_attributes.attr_ndims[self._graph_name_hidden],
                hidden_dim=self.num_channels,
                sub_graph=self._graph_data[(self._graph_name_data, "to", self._graph_name_hidden)],
                src_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
                dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            )

        # Processor hidden -> hidden
        self.processor = instantiate(
            model_config.model.processor,
            _recursive_=False,
            num_channels=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
        )

        # Decoder hidden -> data
        self.decoders = nn.ModuleDict({})
        for target_name in self.target_names:
            self.decoders[target_name] = instantiate(
                model_config.model.decoder,
                _recursive_=False,
                in_channels_src=self.num_channels,
                in_channels_dst=self.node_attributes.attr_ndims[input_name] + self.input_channels[input_name],
                hidden_dim=self.num_channels,
                out_channels_dst=self.target_channels[target_name],
                sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_data)],
                src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                dst_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
            )

        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        # TODO: Bring bounding back
        self.boundings = nn.ModuleList([])
        # self.boundings = nn.ModuleList(
        #    [
        #        instantiate(
        #            cfg,
        #            name_to_index=sample_provider.target.name_to_index,
        #            statistics=sample_provider.input.statistics,
        #            name_to_index_stats=sample_provider.input.name_to_index,
        #        )
        #        for cfg in getattr(model_config.model, "bounding", [])
        #    ]
        # )

    def _assemble_input(self, name: str, x: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        # x.shape: (batch_size, multi_step, ens_dim, grid_size, num_vars)

        # normalize and add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                self.node_attributes(name, batch_size=batch_size),
            ),
            dim=-1,  # feature dimension
        )

        x_skip = x[:, -1, :, :, :]

        return x_data_latent, x_skip

    def _assemble_target(self, name: str, x: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.node_attributes(name, batch_size=batch_size)

    def _disassemble_target(
        self,
        x: torch.Tensor,
        batch_size: int,
        ensemble_size: int,
    ) -> dict[str, torch.Tensor]:
        return (
            einops.rearrange(
                x,
                "(batch ensemble grid) vars -> batch ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
            )
            .to(dtype=torch.float32)
            .clone()
        )

    def _run_mapper(
        self,
        mapper: nn.Module,
        data: tuple[Tensor],
        batch_size: int,
        shard_shapes: tuple[tuple[int, int], tuple[int, int]],
        model_comm_group: Optional[ProcessGroup] = None,
        use_reentrant: bool = False,
    ) -> Tensor:
        """Run mapper with activation checkpoint.

        Parameters
        ----------
        mapper : nn.Module
            Which processor to use
        data : tuple[Tensor]
            tuple of data to pass in
        batch_size: int,
            Batch size
        shard_shapes : tuple[tuple[int, int], tuple[int, int]]
            Shard shapes for the data
        model_comm_group : ProcessGroup
            model communication group, specifies which GPUs work together
            in one model instance
        use_reentrant : bool, optional
            Use reentrant, by default False

        Returns
        -------
        Tensor
            Mapped data
        """
        return checkpoint(
            mapper,
            data,
            batch_size=batch_size,
            shard_shapes=shard_shapes,
            model_comm_group=model_comm_group,
            use_reentrant=use_reentrant,
        )

    def encode(
        self,
        x: dict[str, torch.Tensor],
        shard_shapes_hidden: tuple[list],
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        x_data, x_hidden = x

        x_data_latent, x_hidden_latent, x_data_skip = {}, {}, {}
        for name, x_input_data in x_data.items():
            x_input_data, x_data_skip[name] = self._assemble_input(name, x_input_data, batch_size)

            shard_shapes_input_data = get_shape_shards(x_input_data, 0, model_comm_group)

            x_data_latent[name], x_hidden_latent[name] = self._run_mapper(
                self.encoders[name],
                (x_input_data, x_hidden),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_input_data, shard_shapes_hidden),
                model_comm_group=model_comm_group,
            )

        return x_data_latent, x_hidden_latent, x_data_skip

    def merge_latents(self, latents: dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO: implement different strategies: sum, average, learnable, ...
        return latents["era5"]

    def decode(
        self,
        x: dict[str, torch.Tensor],
        shard_shapes_hidden: tuple[list],
        batch_size: int,
        ensemble_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ):
        x_hidden_latent, x_target_data = x

        x_out = {}
        for name in self.target_names:
            if name in x_target_data:
                x_target_latent = x_target_data[name]
            else:
                x_target_latent = self._assemble_target(name, x_target_data, batch_size)

            shard_shapes_target_data = get_shape_shards(
                x_target_latent, 0, model_comm_group
            )  # This may be passed when name in x_target_data

            x_out[name] = self._run_mapper(
                self.decoders[name],
                (x_hidden_latent, x_target_latent),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hidden, shard_shapes_target_data),
                model_comm_group=model_comm_group,
            )

            x_out[name] = self._disassemble_target(x_out[name], batch_size, ensemble_size)

        return x_out

    def residual_connection(
        self, x: dict[str, torch.Tensor], x_skips: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        y = {}
        for tensor_name, pred in x.items():
            # residual connection (just for the prognostic variables)
            if tensor_name in self.use_residual_connection:
                # TODO: Implement residual connection
                assert False, "Residual Connection NOT IMPLEMENTED yet."
                pred[..., self._internal_output_idx] += x_skip[..., self._internal_input_idx]

            y[tensor_name] = pred

        return y

    def bound(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        y = {}
        for tensor_name, pred in x.items():
            for bounding in self.boundings:
                # bounding performed in the order specified in the config file
                pred = bounding(pred)

            y[tensor_name] = pred

        return y

    def forward(
        self, x: dict[str, Tensor], *, model_comm_group: Optional[ProcessGroup] = None, **kwargs
    ) -> dict[str, Tensor]:
        batch_size = x["era5"].shape[0]
        ensemble_size = 1

        x_hidden = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)
        shard_shapes_hidden = get_shape_shards(x_hidden, 0, model_comm_group)

        x_data_latent, x_hidden_latent, x_data_skip = self.encode(
            (x, x_hidden),
            shard_shapes_hidden=shard_shapes_hidden,
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )

        x_hidden_latent = self.merge_latents(x_hidden_latent)

        x_latent_proc = self.processor(
            x_hidden_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        if self.latent_residual_connection:
            x_latent_proc = x_latent_proc + x_hidden_latent

        x_out = self.decode(
            (x_hidden_latent, x_data_latent),
            shard_shapes_hidden=shard_shapes_hidden,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            model_comm_group=model_comm_group,
        )

        x_out = self.residual_connection(x_out, x_data_skip)
        x_out = self.bound(x_out)
        return x_out
