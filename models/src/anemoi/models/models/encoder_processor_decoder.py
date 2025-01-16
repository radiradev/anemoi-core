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
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiModelEncProcDec(nn.Module):
    """Message passing graph neural network."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        graph_data: HeteroData,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        model_config : DotDict
            Model configuration
        data_indices : dict
            Data indices
        graph_data : HeteroData
            Graph definition
        """
        super().__init__()

        self._graph_data = graph_data
        self._graph_name_data = model_config.graph.data
        self._graph_name_hidden = model_config.graph.hidden

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)
        self.data_indices = data_indices

        self.multi_step = model_config.training.multistep_input
        self.num_channels = model_config.model.num_channels
        self.num_heads_enc = model_config.model.encoder.num_heads
        self.num_heads_dec = model_config.model.decoder.num_heads

        print(
            f"num_heads_dec: {self.num_heads_dec}, hidden_dim: {self.num_channels}, hidden_dim_dec: {self.num_channels // self.num_heads_dec}"
        )

        self.node_attributes = NamedNodesAttributes(model_config.model.trainable_parameters.hidden, self._graph_data)

        input_dim = self.multi_step * self.num_input_channels + self.node_attributes.attr_ndims[self._graph_name_data]

        # Encoder data -> hidden
        self.encoder_list = nn.ModuleList(
            [
                instantiate(
                    model_config.model.encoder,
                    in_channels_src=input_dim,
                    in_channels_dst=self.node_attributes.attr_ndims[self._graph_name_hidden],
                    hidden_dim=self.num_channels // self.num_heads_enc,
                    num_heads=1,
                    sub_graph=self._graph_data[(self._graph_name_data, "to", self._graph_name_hidden)],
                    src_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
                    dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                )
                for _ in range(self.num_heads_enc)
            ]
        )

        # Processor hidden -> hidden
        self.processor = instantiate(
            model_config.model.processor,
            num_channels=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
        )

        # Decoder hidden -> data
        self.decoder_list = nn.ModuleList(
            [
                instantiate(
                    model_config.model.decoder,
                    in_channels_src=self.num_channels // self.num_heads_dec,
                    in_channels_dst=input_dim,
                    hidden_dim=self.num_channels // self.num_heads_dec,
                    num_heads=1,
                    out_channels_dst=self.num_output_channels,
                    sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_data)],
                    src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                    dst_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
                )
                for _ in range(self.num_heads_dec)
            ]
        )

        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        self.boundings = nn.ModuleList(
            [
                instantiate(cfg, name_to_index=self.data_indices.internal_model.output.name_to_index)
                for cfg in getattr(model_config.model, "bounding", [])
            ]
        )

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        self.num_input_channels = len(data_indices.internal_model.input)
        self.num_output_channels = len(data_indices.internal_model.output)
        self._internal_input_idx = data_indices.internal_model.input.prognostic
        self._internal_output_idx = data_indices.internal_model.output.prognostic

    def _assert_matching_indices(self, data_indices: dict) -> None:

        assert len(self._internal_output_idx) == len(data_indices.internal_model.output.full) - len(
            data_indices.internal_model.output.diagnostic
        ), (
            f"Mismatch between the internal data indices ({len(self._internal_output_idx)}) and "
            f"the internal output indices excluding diagnostic variables "
            f"({len(data_indices.internal_model.output.full) - len(data_indices.internal_model.output.diagnostic)})",
        )
        assert len(self._internal_input_idx) == len(
            self._internal_output_idx,
        ), f"Internal model indices must match {self._internal_input_idx} != {self._internal_output_idx}"

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

    def forward(self, x: Tensor, model_comm_group: Optional[ProcessGroup] = None) -> Tensor:
        batch_size = x.shape[0]
        ensemble_size = x.shape[2]

        # add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                self.node_attributes(self._graph_name_data, batch_size=batch_size),
            ),
            dim=-1,  # feature dimension
        )

        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)

        # get shard shapes
        shard_shapes_data = get_shape_shards(x_data_latent, 0, model_comm_group)
        shard_shapes_hidden = get_shape_shards(x_hidden_latent, 0, model_comm_group)

        # Run encoder
        x_latent_list = []
        for i in range(self.num_heads_enc):
            x_data_latent, x_latent_i = self._run_mapper(
                self.encoder_list[i],
                (x_data_latent, x_hidden_latent),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_data, shard_shapes_hidden),
                model_comm_group=model_comm_group,
            )
            x_latent_list.append(x_latent_i)

        x_latent = torch.cat(x_latent_list, dim=-1)

        x_latent_proc = self.processor(
            x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        # add skip connection (hidden -> hidden)
        x_latent_proc = x_latent_proc + x_latent

        # Run decoder
        out_grid_size = self.node_attributes.num_nodes[self._graph_name_data]
        x_out = torch.empty((self.num_heads_dec, out_grid_size, self.num_output_channels), device=x.device)
        x_latent_proc_chunk = torch.tensor_split(x_latent_proc, self.num_heads_dec, dim=-1)
        # TODO: do we also need to chunk x_data_latent?
        print(f"x_latent_proc_chunks: {x_latent_proc_chunk[0].shape}, x_data_latent: {x_data_latent.shape}")
        for i in range(self.num_heads_dec):
            x_out[i] = self._run_mapper(
                self.decoder_list[i],
                (x_latent_proc_chunk[i], x_data_latent),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hidden, shard_shapes_data),  # TODO: check if this is correct with chunking
                model_comm_group=model_comm_group,
            )

        # combine decoder outputs x_out_list by summing them up
        x_out = torch.sum(x_out, dim=0)

        print(f"x_out: {x_out.shape}")

        x_out = (
            einops.rearrange(
                x_out,
                "(batch ensemble grid) vars -> batch ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
            )
            .to(dtype=x.dtype)
            .clone()
        )

        # residual connection (just for the prognostic variables)
        x_out[..., self._internal_output_idx] += x[:, -1, :, :, self._internal_input_idx]

        for bounding in self.boundings:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)

        return x_out
