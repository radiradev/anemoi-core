import einops
import torch
from torch import nn

from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes

# Residual Mapper
# 1. Create sparse matrix from edge index in up/down graphs which should be defined in the config
# 2. Cut out the truncation code from enc-proc-dec model and init residual mapper
# 3. Use the apply truncation as the forward function in ResidualMapper
# 4. Don't process the x_skip in the assemble input, but apply it in the forward function
# 5. Make mapper configurable


class TruncationMapper(nn.Module):
    def __init__(self, graph) -> None:
        super().__init__()

        self.A_down = self._create_sparse_projection_matrix(
            graph["data", "to", "hidden"].edge_index,
            graph["data", "to", "hidden"].edge_attr,
            graph["data"].num_nodes,
            graph["hidden"].num_nodes,
        )

        self.A_up = self._create_sparse_projection_matrix(
            graph["hidden", "to", "data"].edge_index,
            graph["hidden", "to", "data"].edge_attr,
            graph["hidden"].num_nodes,
            graph["data"].num_nodes,
        )

    def forward(self, x, grid_shard_shapes=None, model_comm_group=None):
        batch_size = x.shape[0]

        x = x[:, -1, ...]  # pick current date
        if self.A_down is not None or self.A_up is not None:
            x = einops.rearrange(x, "batch ensemble grid features -> (batch ensemble) grid features")
            x = self._to_channel_shards(x, grid_shard_shapes, model_comm_group)
            x = self._truncate_fields(x)
            x = self._to_grid_shards(x, grid_shard_shapes, model_comm_group)
            x = einops.rearrange(x, "(batch ensemble) grid features -> batch ensemble grid features", batch=batch_size)

        return x

    def _sparse_projection(self, A, x):
        out = []
        for i in range(x.shape[0]):
            out.append(torch.sparse.mm(A.T, x[i, ...]))
        return torch.stack(out)

    def _to_channel_shards(self, x, grid_shard_shapes=None, model_comm_group=None):
        return self._reshard(x, shard_channels, grid_shard_shapes, model_comm_group)

    def _to_grid_shards(self, x, grid_shard_shapes=None, model_comm_group=None):
        return self._reshard(x, gather_channels, grid_shard_shapes, model_comm_group)

    def _reshard(self, x, fn, grid_shard_shapes=None, model_comm_group=None):
        if grid_shard_shapes is not None:
            shard_shapes = self._get_shard_shapes(x, 0, grid_shard_shapes, model_comm_group)
            x = fn(x, shard_shapes, model_comm_group)
        return x

    def _truncate_fields(self, x):
        # A_down and A_up are sparse tensors and cannot be registered as buffers,
        # because DDP does not support broadcasting sparse tensors.
        if self.A_down is not None:
            A_down = self.A_down.to(x.device)
            x = self._sparse_projection(A_down, x)
        if self.A_up is not None:
            A_up = self.A_up.to(x.device)
            x = self._sparse_projection(A_up, x)

        return x

    def _get_shard_shapes(self, x, dim=0, shard_shapes_dim=None, model_comm_group=None):
        if shard_shapes_dim is None:
            return get_shard_shapes(x, dim, model_comm_group)
        else:
            return apply_shard_shapes(x, dim, shard_shapes_dim)

    @staticmethod
    def _create_sparse_projection_matrix(
        edge_index,
        edge_attribute,
        num_source_nodes,
        num_target_nodes,
    ):
        sparse_projection_matrix = torch.sparse_coo_tensor(
            edge_index,
            edge_attribute,
            (num_source_nodes, num_target_nodes),
            device=edge_index.device,
        )

        return sparse_projection_matrix.coalesce()


class SkipConnection(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        x = x[:, -1, ...]  # pick current date
        return x
