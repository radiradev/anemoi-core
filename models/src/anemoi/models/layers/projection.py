# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import torch
from hydra.utils import instantiate
from torch import nn
from torch_geometric.data import HeteroData



class NodeEmbedder(nn.Module):
    def __init__(
        self,
        config,
        num_input_channels: dict[str, int],
        num_output_channels: dict[str, int], 
        sources: dict[str, str],
        **kwargs
    ):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.sources = sources

        self.embedders = nn.ModuleDict(
            {
                in_source: instantiate(
                    config, 
                    _recursive_=False,
                    in_features=num_input_channels[in_source], 
                    out_features=num_output_channels[out_source],
                )
                for in_source, out_source in sources.items()
            }
        )

    def forward(self, x: dict[str, torch.Tensor], **kwargs) -> HeteroData:
        new = {}
        for data_source, encoded_source in self.sources.items():
            new[encoded_source] = self.embedders[data_source](x[data_source])
        
        # input: [{"1": tensor, "1": tensor, "2": tensor, ...}]
        # TODO: x_data_latent = concat_tensor_from_same_source(x_data_latent)
        return new


class NodeProjector(nn.Module):
    """Class to project the node representations."""

    def __init__(
        self,
        config,
        num_input_channels: dict[str, int],
        num_output_channels: dict[str, int], 
        sources: dict[str, str],
        **kwargs
    ):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.sources = sources

        self.projectors = nn.ModuleDict(
            {
                out_source: instantiate(
                    config, 
                    _recursive_=False,
                    in_features=num_input_channels[in_source], 
                    out_features=num_output_channels[out_source],
                )
                for in_source, out_source in sources.items()
            }
        )

    def forward(
        self, x: dict[str, torch.Tensor], slices: dict[str, dict[str, slice]], dim: int = 0
    ) -> dict[str, torch.Tensor]:
        """Projects the tensor into the different datasets/report types.

        Arguments
        ---------
        x : dict[str, torch.Tensor]
            Node embeddings of the shape (num_nodes, num_channels)
        slice : dict[str, dict[str, slice]]
            A mapping of the slices corresponding to each dataset/report type.
            For example,
                {"era": slice(0, 100), "synop": slice(100, num_nodes)}
            will maps the first 100 values to era values and the rest to synop observations.

        Returns
        -------
        dict[str, torch.Tensor]
            It returns a dict of each dataset/report type with tensors of shape (1, num_source_nodes, dim_source_nodes)
        """
        x_data_raws = {}
        for name, x_out in x.items():
            for data_name, node_slice in slices[name].items():
                x_data_raws[data_name] = self.projectors[data_name](x_out[node_slice])
        return x_data_raws
