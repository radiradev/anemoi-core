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

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.shapes import change_channels_in_shape
from anemoi.models.distributed.shapes import get_shape_shards
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict

from anemoi.models.layers.mixer import ChannelMixer

LOGGER = logging.getLogger(__name__)


class AnemoiModelEncProcDec(nn.Module):
    """Message passing graph neural network."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
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
        self.statistics = statistics

        self.multi_step = model_config.training.multistep_input
        self.num_channels = model_config.model.num_channels

        self.num_chunks_enc = model_config.model.encoder.num_chunks
        self.num_chunks_dec = model_config.model.decoder.num_chunks
        self.node_attributes = NamedNodesAttributes(model_config.model.trainable_parameters.hidden, self._graph_data)

        input_dim = self.multi_step * self.num_input_channels + self.node_attributes.attr_ndims[self._graph_name_data]

        # read config.model.layer_kernels to get the implementation for certain layers
        self.layer_kernels = load_layer_kernels(model_config.get("model.layer_kernels", {}))

        # Encoder data -> hidden
        self.encoder = instantiate(
            model_config.model.encoder,
            in_channels_src=input_dim,
            in_channels_dst=self.node_attributes.attr_ndims[self._graph_name_hidden],
            hidden_dim=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_data, "to", self._graph_name_hidden)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            layer_kernels=self.layer_kernels,
        )

        self.encoder_mixer = instantiate(model_config.model.encoder_mixer)
        self.decoder_mixer = instantiate(model_config.model.decoder_mixer)

        # Processor hidden -> hidden
        self.processor = instantiate(
            model_config.model.processor,
            num_channels=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            layer_kernels=self.layer_kernels,
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
            layer_kernels=self.layer_kernels,
        )

        self.node_data_extractor = nn.Sequential(
            nn.LayerNorm(self.num_channels), nn.Linear(self.num_channels, self.num_output_channels)
        )

        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        self.boundings = nn.ModuleList(
            [
                instantiate(
                    cfg,
                    name_to_index=self.data_indices.internal_model.output.name_to_index,
                    statistics=self.statistics,
                    name_to_index_stats=self.data_indices.data.input.name_to_index,
                )
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
        chunk_number: int,
        number_of_chunks: int,
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
            chunk_number=chunk_number,
            number_of_chunks=number_of_chunks,
            shard_shapes=shard_shapes,
            model_comm_group=model_comm_group,
            use_reentrant=use_reentrant,
        )

    def mix_channels_and_extract_features(self, x: Tensor, x_skip, x_ref, shapes_x: tuple, batch_size: int, ensemble_size: int, model_comm_group: Optional[ProcessGroup] = None) -> Tensor:
        """Extracts output features from the output of the decoder.

        Parameters
        ----------
        x : Tensor
            Output of the decoder
        x : Tensor
            Skip connection
        x_ref : Tensor
            Reference data for skipped connection
        batch_size : int
            Batch size
        ensemble_size : int
            Ensemble size
        model_comm_group : Optional[ProcessGroup]
            Model communication group

        Returns
        -------
        Tensor
            Extracted features
        """
        x = self.decoder_mixer(x, x_skip)
        x = self.node_data_extractor(x)
        x = gather_tensor(x, 0, change_channels_in_shape(shapes_x, self.num_output_channels), model_comm_group)

        x = (
            einops.rearrange(
                x,
                "(batch ensemble grid) vars -> batch ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
            )
            .to(dtype=x_ref.dtype)
            .clone()
        )

        # residual connection (just for the prognostic variables)
        x[..., self._internal_output_idx] += x_ref[:, -1, :, :, self._internal_input_idx]

        for bounding in self.boundings:
            # bounding performed in the order specified in the config file
            x = bounding(x)

        return x

    def forward(self, x: Tensor, model_comm_group: Optional[ProcessGroup] = None) -> Tensor:
        batch_size = x.shape[0]
        ensemble_size = x.shape[2]

        # add data positional info (lat/lon)
        x_data = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                self.node_attributes(self._graph_name_data, batch_size=batch_size),
            ),
            dim=-1,  # feature dimension
        )

        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)

        # get shard shapes
        shard_shapes_data = get_shape_shards(x_data, 0, model_comm_group)
        shard_shapes_hidden = get_shape_shards(x_hidden_latent, 0, model_comm_group)
        rank = torch.distributed.get_rank(model_comm_group)
        data_grid_shard_size = shard_shapes_data[rank][0]
        hidden_grid_shard_size = shard_shapes_hidden[rank][0]        

        # Run encoder

        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        else:
            target_dtype = x_data.dtype

        # shape: (grid_shard, vars)
        x_latent = torch.zeros((hidden_grid_shard_size, self.num_channels), dtype=target_dtype, device=x_hidden_latent.device)
        for i in range(self.num_chunks_enc):
            # x_latent_skip, x_data_latent, the same in each iter -> check!, mapper returns None until last iter
            x_data_latent, x_latent_i, x_latent_skip = self._run_mapper(
                self.encoder,
                (x_data, x_hidden_latent),
                batch_size=batch_size,
                chunk_number=i,
                number_of_chunks=self.num_chunks_enc,
                shard_shapes=(shard_shapes_data, shard_shapes_hidden),
                model_comm_group=model_comm_group,
            )
            x_latent = x_latent + x_latent_i

        x_latent = checkpoint(self.encoder_mixer, x_latent, x_latent_skip, use_reentrant=False)

        x_latent_proc = self.processor(
            x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        # add skip connection (hidden -> hidden)
        x_latent_proc = x_latent_proc + x_latent

        # Run decoder
        x_out = torch.zeros((data_grid_shard_size, self.num_channels), dtype=target_dtype, device=x_data.device)
        for i in range(self.num_chunks_dec):
            # x_out_skip is the same in each iter -> check!, mapper returns None until last iter
            x_out_i, x_out_skip = self._run_mapper(
                    self.decoder,
                    (x_latent_proc, x_data),
                    batch_size=batch_size,
                    chunk_number=i,
                    number_of_chunks=self.num_chunks_enc,
                    shard_shapes=(shard_shapes_hidden, shard_shapes_data),
                    model_comm_group=model_comm_group,
            )
            x_out = x_out + x_out_i

        x_out = checkpoint(
            self.mix_channels_and_extract_features,
            x_out,
            x_out_skip, 
            x, 
            shard_shapes_data, 
            batch_size, 
            ensemble_size, 
            model_comm_group, 
            use_reentrant=False
        )

        return x_out
