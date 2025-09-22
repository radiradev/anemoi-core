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
    coords = torch.cat([latitudes, longitudes], dim=0).to(torch.float32)  # shape: (2, num_points)
    return torch.cat([torch.sin(coords), torch.cos(coords)], dim=0)


class NodeEmbedder(nn.Module):
    def __init__(
        self,
        config,
        num_input_channels: dict[str, int],
        num_output_channels: dict[str, int],
        dimensions_order: dict[str, tuple[str, ...]],
        coord_dimension: int = 4,  # sin() cos() of lat and lon
    ):
        super().__init__()
        assert set(num_input_channels.keys()) == set(num_output_channels.keys())
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.dimensions_order = dimensions_order
        for dims in self.dimensions_order.values():
            assert "batch" in dims, "Expected 'batch' in dimensions_order"
            assert "variables" in dims, "Expected 'variables' in dimensions_order"
            assert "values" in dims, "Expected 'values' in dimensions_order"

        self.embedders = nn.ModuleDict(
            {
                key: instantiate(
                    config,
                    _recursive_=False,
                    in_features=num_input_channels[key] + coord_dimension,
                    out_features=num_output_channels[key],
                )
                for key in self.num_input_channels.keys()
            }
        )

    def forward(self, x: dict[str, torch.Tensor], batch_size: int, **kwargs) -> dict[str, torch.Tensor]:
        out = x.new_empty()
        for key, box in x.items():
            data = box["data"]  # shape: (1, num_channels, num_points)
            vars_dim = self.dimensions_order[key].index("variables")

            dims = tuple(str(d) if d in ["variables", "values"] else "1" for d in self.dimensions_order[key])
            sincos_latlons = _get_coords(box["latitudes"], box["longitudes"]) # shape: (4, num_points)
            sincos_latlons = einops.rearrange(sincos_latlons, f"variables values -> {' '.join(dims)}")
            sincos_latlons = sincos_latlons.expand(batch_size, -1, -1)

            data = torch.cat([data, sincos_latlons], dim=vars_dim)

            squash_vars = [d for d in self.dimensions_order[key] if d != "variables"]
            data = einops.rearrange(data, f"{' '.join(self.dimensions_order[key])} -> ({' '.join(squash_vars)}) variables")

            out[key] = self.embedders[key](data) # shape: (num_nodes, num_channels)
        return out


class NodeProjector(nn.Module):
    """Class to project the node representations."""

    def __init__(
        self,
        config,
        num_input_channels: dict[str, int],
        num_output_channels: dict[str, int],
        dimensions_order: dict[str, tuple[str, ...]],
    ):
        super().__init__()
        assert set(num_input_channels.keys()) == set(num_output_channels.keys())
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.dimensions_order = dimensions_order
        for dims in self.dimensions_order.values():
            assert "batch" in dims, "Expected 'batch' in dimensions_order"
            assert "variables" in dims, "Expected 'variables' in dimensions_order"
            assert "values" in dims, "Expected 'values' in dimensions_order"

        self.projectors = nn.ModuleDict(
            {
                source_name: instantiate(
                    config,
                    _recursive_=False,
                    in_features=num_input_channels[source_name],
                    out_features=num_output_channels[source_name],
                )
                for source_name in num_input_channels.keys()
            }
        )

    def forward(self, x: dict[str, torch.Tensor], batch_size: int) -> dict[str, torch.Tensor]:
        """Projects the tensor into the different datasets/report types.

        Arguments
        ---------
        x : dict[str, torch.Tensor]
            Node embeddings of the shape (num_nodes, num_channels)

        Returns
        -------
        dict[str, torch.Tensor]
            It returns a dict of each dataset/report type with tensors of shape (1, num_source_nodes, dim_source_nodes)
        """
        out = x.new_empty()
        for name, data in x.items():
            squashed_vars = [d for d in self.dimensions_order[name] if d != "variables"]
            out[name] = einops.rearrange(
                self.projectors[name](data),
                f"({' '.join(squashed_vars)}) variables -> {' '.join(self.dimensions_order[name])}", 
                batch=batch_size,
            )
        return out