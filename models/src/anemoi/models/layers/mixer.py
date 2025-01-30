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
        channels: int,
        mlp_hidden_ratio: int = 4,
        activation: str = "GELU",
        num_chunks: int = 1,
        **kwargs,
    ) -> None:
        """Initialize ChannelMixerMapper.

        Parameters
        ----------
        channels : int
            Number of channels.
        mlp_hidden_ratio : int,
            factor for hidden mlp dimension
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
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * mlp_hidden_ratio),
            act_func(),
            nn.Linear(channels * mlp_hidden_ratio, channels),
        )

    def forward(
        self,
        x: Tensor,
        x_skip: Tensor,
    ):
        num_chunks = self.num_chunks if self.training else NUM_CHUNKS_INFERENCE

        nodes = x + x_skip # skipped connection after attention

        # compute nodes_new_dst = self.node_dst_mlp(out) + out in chunks:
        nodes_new_dst = torch.cat(
            [self.node_dst_mlp(chunk) + chunk for chunk in nodes.tensor_split(num_chunks, dim=0)], dim=0
        )

        return nodes_new_dst
