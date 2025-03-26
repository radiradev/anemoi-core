# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
import math
import copy
from typing import Callable, Optional, Tuple, Union

import einops
import torch
from torch import Tensor, nn
try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask
except ImportError:
    raise ImportError("Error: Flex attention not available in your PyTorch version. Please update PyTorch.")

from torch_geometric.data import HeteroData
from omegaconf import ListConfig

LOGGER = logging.getLogger(__name__)


def calculate_scaled_attention_attention_spans(
    base_attention_span,
    base_grid_name,
    target_grid_name,
    scaling_method,
    _graph_data,
    method,
):
    """
    Calculate scaled attention spans based on grid sizes.
    
    Parameters
    ----------
    base_attention_span : int or list
        Base attention span value
    base_grid_name : str
        Name of the base grid
    target_grid_name : str
        Name of the target grid
    scaling_method : str
        Method to use for scaling
    _graph_data : HeteroData
        Graph data containing grid information
    method : str
        Attention method
        
    Returns
    -------
    int or list
        Scaled attention span
    """
    # Simple pass-through if no scaling needed
    if base_grid_name is None or base_grid_name == target_grid_name:
        return base_attention_span
    
    # Scale based on grid sizes
    if scaling_method == "constant_span_relative_to_grid_size":
        base_grid_size = _graph_data[base_grid_name].num_nodes
        target_grid_size = _graph_data[target_grid_name].num_nodes
        
        scale_factor = (target_grid_size / base_grid_size) ** 0.5
        
        if isinstance(base_attention_span, (list, tuple, ListConfig)):
            return [int(span * scale_factor) for span in base_attention_span]
        else:
            return int(base_attention_span * scale_factor)
    
    # Add other scaling methods as needed
    
    # Default: return unmodified
    return base_attention_span


class BlockMaskManager(nn.Module):
    """
    Manages block masks for flex attention by creating and caching them for different devices.
    
    The class handles different methods for determining attention patterns:
    - window: Simple window-based attention
    - oval_radius_lat_bands_lon_idx: Uses latitude bands and longitude indices
    - oval_radius_distance: Uses geographical distance calculations
    """

    scaling_method = {
        "window": "constant_span_relative_to_grid_size",
        "oval_radius_lat_bands_lon_idx": "constant_span_relative_to_grid_size",
        "oval_radius_distance": "constant_span_relative_to_grid_size",
    }

    def __init__(
        self,
        graph: HeteroData,
        query_grid_name: str,
        keyvalue_grid_name: str,
        attention_span: Optional[int] = None,
        base_attention_span: Optional[int] = None,
        method: str = "oval_radius_distance",
        base_grid: str = "query",
        block_size: Union[int, Tuple[int, int]] = 32,  # Default block size
        bmcachedtype: torch.dtype | None = None,
        **kwargs,
    ):
        """
        Initialize BlockMaskManager.
        
        Parameters
        ----------
        graph : HeteroData
            Graph data containing grid information
        query_grid_name : str
            Name of the query grid
        keyvalue_grid_name : str
            Name of the key-value grid
        attention_span : Optional[int], optional
            Attention span, by default None
        base_attention_span : Optional[int], optional
            Base attention span for scaling, by default None
        method : str, optional
            Method for attention pattern, by default "oval_radius_distance"
        base_grid : str, optional
            Base grid for scaling, by default "query"
        block_size : Union[int, Tuple[int, int]], optional
            Block size for block mask, by default 32
        bmcachedtype : torch.dtype | None, optional
            Data type for cached calculations, by default None
        """
        super().__init__()

        self.graph = graph
        self.map_device_block_mask: dict[torch.device, BlockMask] = {}
        self.bmcachedtype = bmcachedtype

        assert method in [
            "window",
            "oval_radius_lat_bands_lon_idx",
            "oval_radius_distance",
        ], f"Method {method} not supported"
        self.method = method

        assert attention_span is not None or base_attention_span is not None
        self.setup_attention_span(
            attention_span,
            base_attention_span,
            base_grid_name=kwargs.get("base_attention_span_grid", None),
            query_grid_name=query_grid_name,
            keyvalue_grid_name=keyvalue_grid_name,
            base_grid=base_grid,
            **kwargs,
        )

        self.base_grid = base_grid
        self.query_grid_name = query_grid_name
        self.keyvalue_grid_name = keyvalue_grid_name
        self.block_size = block_size
        
        if isinstance(block_size, ListConfig):
            self.block_size = list(block_size)

        self.setup_attn_mask()
        self.setup_mask_mod()

    def setup_attention_span(
        self,
        attention_span=None,
        base_attention_span=None,
        base_grid_name=None,
        query_grid_name=None,
        keyvalue_grid_name=None,
        base_grid=None,
        **kwargs,
    ):
        """
        Setup attention span, potentially scaling based on grid sizes.
        
        Parameters
        ----------
        attention_span : int or list, optional
            Attention span, by default None
        base_attention_span : int or list, optional
            Base attention span for scaling, by default None
        base_grid_name : str, optional
            Name of the base grid, by default None
        query_grid_name : str, optional
            Name of the query grid, by default None
        keyvalue_grid_name : str, optional
            Name of the key-value grid, by default None
        base_grid : str, optional
            Base grid for scaling, by default None
        """
        if attention_span is not None:
            pass
        else:
            attention_span = calculate_scaled_attention_attention_spans(
                base_attention_span,
                base_grid_name=kwargs.get("base_attention_span_grid", None),
                target_grid_name=query_grid_name if base_grid == "query" else keyvalue_grid_name,
                scaling_method=self.scaling_method[self.method],
                _graph_data=self.graph,
                method=self.method,
            )

        self.attention_span = tuple(attention_span) if isinstance(attention_span, list) else attention_span

    def setup_attn_mask(self):
        """Setup attention mask parameters based on the chosen method."""
        if self.method == "window":
            self.attention_radius = copy.deepcopy(self.attention_span)
            
        elif self.method in ["oval_radius_lat_bands_lon_idx", "oval_radius_distance"]:
            assert (
                self.bmcachedtype is None or self.bmcachedtype == torch.float32
            ), "Precision required for this to work is float32"

            if self.method == "oval_radius_lat_bands_lon_idx":
                self.attention_radius_lat = copy.deepcopy(self.attention_span[0])
                self.attention_radius_lon = copy.deepcopy(self.attention_span[1])
                
            elif self.method == "oval_radius_distance":
                self.attention_radius_meridian = copy.deepcopy(self.attention_span[0])
                self.attention_radius_zonal = copy.deepcopy(self.attention_span[1])

                # Constants for geographical calculations
                self.distance_between_lat_degrees_km = 111.0  # km
                self.two_pi = 2 * torch.pi
                self.pi_over_2 = torch.pi / 2
                self.earth_radius_km = 6371.0
                self.earth_radius_km_squared = self.earth_radius_km * self.earth_radius_km

                # Determining grid parameters
                self.angle_between_lat_bands_q = torch.pi / (
                    self.get_grid_latlon(self.query_grid_name)[:, 0].unique().shape[0]
                )

                if self.query_grid_name == self.keyvalue_grid_name:
                    self.angle_between_lat_bands_kv = self.angle_between_lat_bands_q
                else:
                    self.angle_between_lat_bands_kv = torch.pi / (
                        self.get_grid_latlon(self.keyvalue_grid_name)[:, 0].unique().shape[0]
                    )

            # Precomputing Query Grid Parameters
            self.lat_bands_div2_q = self.get_grid_latlon(self.query_grid_name)[:, 0].unique().shape[0] // 2
            self.lat_band_start_idx_q = self.get_latitude_band_start_idxs(
                self.get_grid_latlon(self.query_grid_name)
            )
            self.southern_hemisphere_start_idx_q = self.lat_band_start_idx_q[(self.lat_bands_div2_q - 1) + 1]

            # Determining Keyvalue Grid Parameters
            if self.query_grid_name == self.keyvalue_grid_name:
                self.lat_bands_div2_kv = self.lat_bands_div2_q
                self.southern_hemisphere_start_idx_kv = self.southern_hemisphere_start_idx_q
                self.lat_band_start_idx_kv = self.lat_band_start_idx_q
            else:
                self.lat_bands_div2_kv = self.get_grid_latlon(self.keyvalue_grid_name)[:, 0].unique().shape[0] // 2
                self.lat_band_start_idx_kv = self.get_latitude_band_start_idxs(
                    self.get_grid_latlon(self.keyvalue_grid_name)
                )
                self.southern_hemisphere_start_idx_kv = self.lat_band_start_idx_kv[(self.lat_bands_div2_kv - 1) + 1]
        else:
            raise ValueError(f"Method {self.method} not supported")

    def get_grid_latlon(self, grid_name: str):
        """Get latitude and longitude from grid.
        
        Parameters
        ----------
        grid_name : str
            Name of the grid
            
        Returns
        -------
        Tensor
            Tensor of shape (grid_size, 2) with latitude and longitude coordinates
        """
        return self.graph[grid_name].x  # (grid_size, 2)

    def get_latitude_band_start_idxs(self, grid_latlon: torch.Tensor, tol: float = 1e-5):
        """
        Get the start indices for each latitude band.
        
        Parameters
        ----------
        grid_latlon : torch.Tensor
            Grid latitude and longitude tensor of shape (grid_size, 2)
        tol : float, optional
            Tolerance for latitude differences, by default 1e-5
            
        Returns
        -------
        list
            List of start indices for each latitude band
        """
        lat_list = grid_latlon[:, 0].tolist()
        lon_band_start_idx = [0]
        N = len(lat_list)
        
        for i in range(1, N):
            # If latitude changes beyond tolerance, then a new row starts
            if abs(lat_list[i] - lat_list[i - 1]) > tol:
                lon_band_start_idx.append(i)

        lon_band_start_idx.append(N + 1)
        return lon_band_start_idx

    def setup_mask_mod(self):
        """Set up the appropriate mask modification function based on the method."""
        if self.method == "window":
            self.mask_mod = self.mask_mod_window
            
        elif self.method == "oval_radius_lat_bands_lon_idx":
            if self.base_grid == "query":
                self.mask_mod = self.mask_mod_oval_radius_lat_bands_lon_idx_base_grid_query
            elif self.base_grid == "keyvalue":
                self.mask_mod = self.mask_mod_oval_radius_lat_bands_lon_idx_base_grid_keyvalue
                
        elif self.method == "oval_radius_distance":
            self.mask_mod = self.mask_mod_oval_radius_distance
            
        else:
            raise ValueError(f"Invalid method: {self.method} with base grid: {self.base_grid}")

    def get_block_mask(self, device: torch.device):
        """
        Get block mask for the given device, creating it if necessary.
        
        Parameters
        ----------
        device : torch.device
            Device for the block mask
            
        Returns
        -------
        BlockMask
            Block mask for the device
        """
        if device in self.map_device_block_mask:
            return self.map_device_block_mask[device]
        else:
            q_grid_size = self.graph[self.query_grid_name].num_nodes
            kv_grid_size = self.graph[self.keyvalue_grid_name].num_nodes

            block_mask = create_block_mask(
                self.mask_mod,
                B=None,
                H=None,
                Q_LEN=q_grid_size,
                KV_LEN=kv_grid_size,
                device=device,
                BLOCK_SIZE=self.block_size,
            )
            self.map_device_block_mask[device] = block_mask
            return block_mask

    def mask_mod_window(self, b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        """
        Window-based mask modification function.
        
        Parameters
        ----------
        b : Tensor
            Batch index
        h : Tensor
            Head index
        q_idx : Tensor
            Query index
        kv_idx : Tensor
            Key-value index
            
        Returns
        -------
        Tensor
            Boolean tensor indicating which positions are in the window
        """
        distance = torch.abs(q_idx - kv_idx)
        return distance <= self.attention_radius

    def mask_mod_oval_radius_lat_bands_lon_idx_base_grid_query(
        self, b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Elliptical window attention for latitude-longitude grids (query-centric).
        
        Parameters
        ----------
        b : torch.Tensor
            Batch index
        h : torch.Tensor
            Head index
        q_idx : torch.Tensor
            Query index
        kv_idx : torch.Tensor
            Key-value index
            
        Returns
        -------
        torch.Tensor
            Boolean tensor indicating which positions are in the elliptical window
        """
        max_lat_radius = self.attention_radius_lat
        max_lon_radius = self.attention_radius_lon

        q_lat_band, q_lon_idx = self.decode_latlon(
            q_idx.to(self.bmcachedtype), self.southern_hemisphere_start_idx_q, self.lat_bands_div2_q
        )
        kv_lat_band, kv_lon_idx = self.decode_latlon(
            kv_idx.to(self.bmcachedtype), self.southern_hemisphere_start_idx_kv, self.lat_bands_div2_kv
        )

        q_nodes_per_band = self.nodes_per_lat_band(q_lat_band, self.lat_bands_div2_q)
        kv_nodes_per_band = self.nodes_per_lat_band(kv_lat_band, self.lat_bands_div2_kv)

        kv_proj_in_q_lon_idx = (kv_lon_idx / kv_nodes_per_band) * q_nodes_per_band

        lat_diff = q_lat_band - kv_lat_band
        lon_diff = torch.abs(q_lon_idx - kv_proj_in_q_lon_idx)
        lon_diff = torch.where(lon_diff > (q_nodes_per_band // 2), q_nodes_per_band - lon_diff, lon_diff)

        in_oval = (lat_diff**2 / max_lat_radius**2 + lon_diff**2 / max_lon_radius**2) <= 1
        return in_oval

    def mask_mod_oval_radius_lat_bands_lon_idx_base_grid_keyvalue(
        self, b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Elliptical window attention for latitude-longitude grids (key-centric).
        
        Parameters
        ----------
        b : torch.Tensor
            Batch index
        h : torch.Tensor
            Head index
        q_idx : torch.Tensor
            Query index
        kv_idx : torch.Tensor
            Key-value index
            
        Returns
        -------
        torch.Tensor
            Boolean tensor indicating which positions are in the elliptical window
        """
        max_lat_radius = self.attention_radius_lat
        max_lon_radius = self.attention_radius_lon

        q_lat_band, q_lon_idx = self.decode_latlon(
            q_idx.to(self.bmcachedtype), self.southern_hemisphere_start_idx_q, self.lat_bands_div2_q
        )
        kv_lat_band, kv_lon_idx = self.decode_latlon(
            kv_idx.to(self.bmcachedtype), self.southern_hemisphere_start_idx_kv, self.lat_bands_div2_kv
        )

        q_nodes_per_band = self.nodes_per_lat_band(q_lat_band, self.lat_bands_div2_q)
        kv_nodes_per_band = self.nodes_per_lat_band(kv_lat_band, self.lat_bands_div2_kv)

        q_proj_in_kv_lon_idx = (q_lon_idx / q_nodes_per_band) * kv_nodes_per_band

        lat_diff = q_lat_band - kv_lat_band
        lon_diff = torch.abs(kv_lon_idx - q_proj_in_kv_lon_idx)
        lon_diff = torch.where(lon_diff > (kv_nodes_per_band // 2), kv_nodes_per_band - lon_diff, lon_diff)

        in_oval = (lat_diff**2 / max_lat_radius**2 + lon_diff**2 / max_lon_radius**2) <= 1
        return in_oval

    def mask_mod_oval_radius_distance(
        self, b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Elliptical window attention using geographic distance calculations.
        
        Parameters
        ----------
        b : torch.Tensor
            Batch index
        h : torch.Tensor
            Head index
        q_idx : torch.Tensor
            Query index
        kv_idx : torch.Tensor
            Key-value index
            
        Returns
        -------
        torch.Tensor
            Boolean tensor indicating which positions are in the elliptical window based on distance
        """
        max_meridian_radius = self.attention_radius_meridian
        max_zonal_radius = self.attention_radius_zonal

        q_lat_band, q_lon_idx = self.decode_latlon(
            q_idx.to(self.bmcachedtype), self.southern_hemisphere_start_idx_q, self.lat_bands_div2_q
        )
        kv_lat_band, kv_lon_idx = self.decode_latlon(
            kv_idx.to(self.bmcachedtype), self.southern_hemisphere_start_idx_kv, self.lat_bands_div2_kv
        )

        q_lat_radians = self.pi_over_2 - (q_lat_band + 1) * self.angle_between_lat_bands_q
        kv_lat_radians = self.pi_over_2 - (kv_lat_band + 1) * self.angle_between_lat_bands_kv
        lat_diff_in_radians = q_lat_radians - kv_lat_radians

        q_nodes = self.nodes_per_lat_band(q_lat_band, self.lat_bands_div2_q)
        kv_nodes = self.nodes_per_lat_band(kv_lat_band, self.lat_bands_div2_kv)

        q_norm = q_lon_idx / q_nodes
        kv_norm = kv_lon_idx / kv_nodes

        lon_norm_diff = self.two_pi * torch.abs(q_norm - kv_norm)
        lon_diff_in_radians = torch.where(lon_norm_diff > torch.pi, self.two_pi - lon_norm_diff, lon_norm_diff)

        haversine_distance_squared = self.approx_equirectangular_arithmetic_mean_squared(
            dlat=lat_diff_in_radians, dlon=lon_diff_in_radians, lat1=q_lat_radians, lat2=kv_lat_radians
        )

        theta = torch.atan2(lon_diff_in_radians, lat_diff_in_radians)
        sin_theta = torch.sin(theta)
        sin_theta_squared = sin_theta**2

        in_oval = (
            (1 - sin_theta_squared) * (haversine_distance_squared / max_meridian_radius**2)
            + sin_theta_squared * (haversine_distance_squared / max_zonal_radius**2)
        ) <= 1
        return in_oval

    def decode_latlon(self, idx: torch.Tensor, southern_hemisphere_start_idx: int, lat_bands_div2: int) -> tuple:
        """
        Decode a flattened grid index into latitude band and longitude index.
        
        Parameters
        ----------
        idx : torch.Tensor
            Flattened grid index
        southern_hemisphere_start_idx : int
            Index where southern hemisphere starts
        lat_bands_div2 : int
            Half the number of latitude bands
            
        Returns
        -------
        tuple
            Tuple of (latitude band, longitude index)
        """
        north_mask = idx < southern_hemisphere_start_idx

        # Northern hemisphere calculation
        r_n = torch.floor((-18.0 + torch.sqrt(324.0 + 8.0 * idx)) * 0.25)
        lon_n = idx - (2 * r_n**2 + 18 * r_n)

        # Southern hemisphere calculation
        idx_s = idx - southern_hemisphere_start_idx
        A = 2.0
        B = 4.0 * lat_bands_div2 + 18.0
        r_s = torch.floor((B - torch.sqrt(B * B - 8.0 * idx_s)) / (2.0 * A))
        lon_s = idx_s - ((4.0 * lat_bands_div2 + 18.0) * r_s - 2.0 * r_s**2)
        lat_band_s = lat_bands_div2 + r_s

        # Combine results
        lat_band_idx = torch.where(north_mask, r_n, lat_band_s)
        lon_idx = torch.where(north_mask, lon_n, lon_s)
        return lat_band_idx, lon_idx

    def nodes_per_lat_band(self, lat_band_idx: torch.Tensor, lat_bands_div2: int) -> torch.Tensor:
        """
        Compute the number of nodes in a given latitude band.
        
        Parameters
        ----------
        lat_band_idx : torch.Tensor
            Latitude band index
        lat_bands_div2 : int
            Half the number of latitude bands
            
        Returns
        -------
        torch.Tensor
            Number of nodes per latitude band
        """
        north_mask = lat_band_idx < lat_bands_div2
        nodes_north = 4.0 * lat_band_idx + 20.0
        nodes_south = 8.0 * lat_bands_div2 - 4.0 * lat_band_idx + 16.0

        return torch.where(north_mask, nodes_north, nodes_south)

    def approx_equirectangular_arithmetic_mean_squared(self, dlat, dlon, lat1, lat2):
        """
        Equirectangular approximation using arithmetic mean.
        
        Parameters
        ----------
        dlat : torch.Tensor
            Latitude difference in radians
        dlon : torch.Tensor
            Longitude difference in radians
        lat1 : torch.Tensor
            First latitude in radians
        lat2 : torch.Tensor
            Second latitude in radians
            
        Returns
        -------
        torch.Tensor
            Squared distance
        """
        cos_mean = torch.cos((lat1 + lat2) / 2)
        x = dlon * cos_mean
        y = dlat
        return self.earth_radius_km_squared * ((x * x) + (y * y))

    def signature(self) -> tuple:
        """Get a signature tuple that uniquely identifies this configuration.
        
        Returns
        -------
        tuple
            Tuple of (attention_span, keyvalue_grid_name, query_grid_name, base_grid)
        """
        return (self.attention_span, self.keyvalue_grid_name, self.query_grid_name, self.base_grid)
