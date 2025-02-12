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
from anemoi.models.layers.embedder import VerticalInformationEmbedder
from anemoi.models.layers.attention import MultiHeadSelfAttention
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

        self.node_attributes = NamedNodesAttributes(model_config.model.trainable_parameters.hidden, self._graph_data)
        
        self.num_levels = model_config.training.vertical_embeddings.num_levels
        self.level_shuffle = model_config.training.vertical_embeddings.level_shuffle
        self.vertical_embeddings_method = model_config.training.vertical_embeddings.method

        #! FOURIER AND ENCODED DIMENSIONS FOR PRESSURE LEVELS
        self.fourier_dim = model_config.training.vertical_embeddings.fourier_dim # small otherwise CUDA OOM
        self.hidden_dim = model_config.training.vertical_embeddings.hidden_dim
        self.encoded_dim = model_config.training.vertical_embeddings.encoded_dim # small otherwise CUDA OOM
        self.num_levels = model_config.training.vertical_embeddings.num_levels

        self.embedder = VerticalInformationEmbedder(level_shuffle=self.level_shuffle, method=self.vertical_embeddings_method, 
                                                                    fourier_dim=self.fourier_dim, hidden_dim=self.hidden_dim,
                                                                    encoded_dim=self.encoded_dim, num_levels=self.num_levels, 
                                                                    data_indices=self.data_indices)

        if self.vertical_embeddings_method == 'concat':
            input_dim = (
                self.multi_step * (self.num_input_channels + self.num_input_channels * self.embedder.encoded_dim)
                + self.node_attributes.attr_ndims[self._graph_name_data]
            )
        # elif self.vertical_embeddings_method == 'attention':
        #     input_dim = (
        #         self.multi_step * (6 * self.embedder.encoded_dim)
        #         + self.node_attributes.attr_ndims[self._graph_name_data]
        #     )
        else:
            input_dim = (
                self.multi_step * (self.num_input_channels)
                + self.node_attributes.attr_ndims[self._graph_name_data]
            )

        # Encoder data -> hidden
        self.encoder = instantiate(
            model_config.model.encoder,
            in_channels_src=input_dim,
            in_channels_dst=self.node_attributes.attr_ndims[self._graph_name_hidden],
            hidden_dim=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_data, "to", self._graph_name_hidden)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
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
        self.decoder = instantiate(
            model_config.model.decoder,
            in_channels_src=self.num_channels,
            in_channels_dst=input_dim,
            hidden_dim=self.num_channels,
            out_channels_dst=self.num_output_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_data)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
        )

        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        self.boundings = nn.ModuleList(
            [
                instantiate(cfg, name_to_index=self.data_indices.internal_model.output.name_to_index)
                for cfg in getattr(model_config.model, "bounding", [])
            ]
        )

        act_func = nn.ReLU()
        self.mlp_mc = nn.Sequential(nn.Linear(self.embedder.num_levels, self.embedder.hidden_dim), act_func)
        self.mlp_mc.append(nn.Linear(self.embedder.hidden_dim, self.embedder.num_levels))
        self.mlp_mc.append(act_func)   

        self.mlp_mc = self.mlp_mc.cuda()   

        self.mc_attention_layer = MultiHeadSelfAttention(
            num_heads=13, # when 1 this is self attention - 8 heads is a balanced choice for most tasks.
            embed_dim= self.embedder.num_levels,
            window_size=26,
            bias=False,
            is_causal=False,
            dropout_p=0,
        ).cuda()          
        self.mc_layer_norm1 = nn.LayerNorm(self.embedder.num_levels).cuda()
        self.mc_layer_norm2 = nn.LayerNorm(self.embedder.num_levels).cuda()        

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

        n_times = x.shape[1]
        num_grid_points = x.shape[3]        
        self.num_variables = int(x.shape[4]/self.num_levels)
        x_reshaped = torch.reshape(x, (batch_size, n_times, ensemble_size, num_grid_points, self.num_variables, self.num_levels)).to("cuda")
        mapped_features = self.embedder(x)


        x_reshaped = x_reshaped + mapped_features

        x_reshaped = einops.rearrange(
                x_reshaped, "batch time ensemble grid vars levels -> batch ensemble grid vars time levels"
            )
        
        x_att = einops.rearrange(
                x_reshaped, "batch ensemble grid vars time levels -> (batch ensemble grid vars time) (levels)"
            )


        x_att = x_att + self.mc_attention_layer(self.mc_layer_norm1(x_att), shapes=[[x_att.shape[0], x_att.shape[1]]], batch_size=240)
        x_att = x_att + self.mlp_mc(self.mc_layer_norm2(x_att))

        x_data_vertical_latent = x_att.reshape(int(x_att.shape[0]/(2*6)), (6*2*self.embedder.num_levels))

        # add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                x_data_vertical_latent,
                self.node_attributes(self._graph_name_data, batch_size=batch_size),
            ),
            dim=-1,  # feature dimension
        )

        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)
        # get shard shapes
        shard_shapes_data = get_shape_shards(x_data_latent, 0, model_comm_group)
        shard_shapes_hidden = get_shape_shards(x_hidden_latent, 0, model_comm_group)

        # Run encoder
        x_data_latent, x_latent = self._run_mapper(
            self.encoder,
            (x_data_latent, x_hidden_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
        )

        x_latent_proc = self.processor(
            x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        # add skip connection (hidden -> hidden)
        x_latent_proc = x_latent_proc + x_latent

        # Run decoder
        x_out = self._run_mapper(
            self.decoder,
            (x_latent_proc, x_data_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
        )

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

        x_out_reshape = torch.reshape(x_out, (batch_size, x_out.shape[1], x_out.shape[2], 6, 13)) #self.embedder.num_variables, self.num_levels))
        if self.level_shuffle:
            x_reorder = x_out_reshape[..., self.embedder.rand_rev]
        else:
            x_reorder = x_out_reshape

        x_out = (
            einops.rearrange(
                x_reorder,
                "batch ensemble grid vars level -> batch ensemble grid (vars level)",
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
