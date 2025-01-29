# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os

import torch
from torch import Tensor
from torch import nn
from torch_geometric.typing import PairTensor

LOGGER = logging.getLogger(__name__)

# Number of Mapper chunks used during inference (https://github.com/ecmwf/anemoi-models/pull/46)
NUM_CHUNKS_INFERENCE = int(os.environ.get("ANEMOI_INFERENCE_NUM_CHUNKS", "1"))

class ChannelMixer(nn.Module):
    """Graph Transformer Block for node embeddings."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        activation: str = "GELU",
        num_chunks: int = 1,
        **kwargs,
    ) -> None:
        """Initialize ChannelMixerMapper.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        hidden_dim : int,
            hidden mlp dimension
        out_channels : int
            Number of output channels.
        activation : str, optional
            Activation function, by default "GELU"
        update_src_nodes: bool, by default False
            Update src if src and dst nodes are given
        """
        super().__init__(**kwargs)

        self.num_chunks = num_chunks

        try:
            act_func = getattr(nn, activation)
        except AttributeError as ae:
            LOGGER.error("Activation function %s not supported", activation)
            raise RuntimeError from ae

        self.node_dst_mlp = nn.Sequential(
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, hidden_dim),
            act_func(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(
        self,
        x: PairTensor,
    ):
        num_chunks = self.num_chunks if self.training else NUM_CHUNKS_INFERENCE

        # fuse inputs, todo
        nodes = torch.cat(x, dim=-1)

        # compute nodes_new_dst = self.node_dst_mlp(out) + out in chunks:
        nodes_new_dst = torch.cat(
            [self.node_dst_mlp(chunk) + chunk for chunk in nodes.tensor_split(num_chunks, dim=0)], dim=0
        )

        return nodes_new_dst

# todo when updating src nodes as well..
class ChannelMixerMapper(nn.Module):
    """Graph Transformer Block for node embeddings."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        activation: str = "GELU",
        num_chunks: int = 1,
        update_src_nodes: bool = False,
        **kwargs,
    ) -> None:
        """Initialize ChannelMixerMapper.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        hidden_dim : int,
            hidden mlp dimension
        out_channels : int
            Number of output channels.
        activation : str, optional
            Activation function, by default "GELU"
        update_src_nodes: bool, by default False
            Update src if src and dst nodes are given
        """
        super().__init__(**kwargs)

        self.update_src_nodes = update_src_nodes
        self.num_chunks = num_chunks

        try:
            act_func = getattr(nn, activation)
        except AttributeError as ae:
            LOGGER.error("Activation function %s not supported", activation)
            raise RuntimeError from ae

        self.node_dst_mlp = nn.Sequential(
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, hidden_dim),
            act_func(),
            nn.Linear(hidden_dim, out_channels),
        )

        self.layer_norm1 = nn.LayerNorm(in_channels)

        if self.update_src_nodes:
            self.node_src_mlp = nn.Sequential(
                nn.LayerNorm(out_channels),
                nn.Linear(out_channels, hidden_dim),
                act_func(),
                nn.Linear(hidden_dim, out_channels),
            )

    def forward(
        self,
        x: PairTensor,
    ):
        num_chunks = self.num_chunks if self.training else NUM_CHUNKS_INFERENCE

        nodes = x

        # fuse inputs, todo

        # compute nodes_new_dst = self.node_dst_mlp(out) + out in chunks:
        nodes_new_dst = torch.cat(
            [self.node_dst_mlp(chunk) + chunk for chunk in nodes[1].tensor_split(num_chunks, dim=0)], dim=0
        )

        if self.update_src_nodes:
            # compute nodes_new_src = self.node_src_mlp(out) + out in chunks:
            nodes_new_src = torch.cat(
                [self.node_src_mlp(chunk) + chunk for chunk in nodes[0].tensor_split(num_chunks, dim=0)], dim=0
            )
        else:
            nodes_new_src = nodes[0]

        nodes_new = (nodes_new_src, nodes_new_dst)

        return nodes_new
