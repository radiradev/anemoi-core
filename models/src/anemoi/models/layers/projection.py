# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import torch
import einops
from hydra.utils import instantiate
from torch import nn
from torch_geometric.data import HeteroData

def _get_coords(latitudes: torch.Tensor, longitudes: torch.Tensor) -> torch.Tensor:
    coords = torch.cat([latitudes, longitudes], dim=0)
    return torch.cat([torch.sin(coords), torch.cos(coords)], dim=0)


class NodeEmbedder(nn.Module):
    def __init__(
        self,
        config,
        num_input_channels: dict[str, int],
        num_output_channels: dict[str, int],
        coord_dimension: int = 4,  # sin() cos() of lat and lon
        **kwargs
    ):
        super().__init__()
        assert set(num_input_channels.keys()) == set(num_output_channels.keys())
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels

        self.embedders = nn.ModuleDict(
            {
                key: nn.Linear(
                    in_features=num_input_channels[key] + coord_dimension,
                    out_features=num_output_channels[key],
                )
                for key in self.num_input_channels.keys()
            }
        )

    def forward(self, x: dict[str, torch.Tensor], **kwargs) -> HeteroData:
        # input: [{"1": tensor, "1": tensor, "2": tensor, ...}]
        # TODO: x_data_latent = concat_tensor_from_same_source(x_data_latent)
        out = x.new_empty()
        for key, box in x.items():
            data = box["data"]  # shape: (1, num_channels, num_points)
            assert box["dimensions_order"] == ["ensemble", "variables", "values"], "Expected dimensions_order to be ['ensemble', 'variables', 'values']"
            sincos_latlons = self._get_coords(box["latitudes"], box["longitudes"]) # shape: (4, num_points)
            assert sincos_latlons.shape == (4, data.shape[2]), f"sincos_latlons shape {sincos_latlons.shape} does not match expected (4, {data.shape[2]})"
            sincos_latlons = einops.rearrange(sincos_latlons, "coords grid -> 1 coords grid")
            data = torch.cat([data, sincos_latlons], dim=1)
            data = einops.rearrange(data, "1 vars grid -> grid vars")
            out[key] = self.embedders[key](data)
        return out


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
