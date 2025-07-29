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
    def __init__(self, A_down: torch.Tensor, A_up: torch.Tensor = None) -> None:
        super().__init__()

    def forward(self, x):
        return self._apply_truncation(x)

    def _truncate_fields(self, x, A, batch_size=None, auto_cast=False):
        if not batch_size:
            batch_size = x.shape[0]
        out = []
        with torch.amp.autocast(device_type="cuda", enabled=auto_cast):
            for i in range(batch_size):
                out.append(self.A_down.mm(x))
        return torch.stack(out)

    def _get_shard_shapes(self, x, dim=0, shard_shapes_dim=None, model_comm_group=None):
        if shard_shapes_dim is None:
            return get_shard_shapes(x, dim, model_comm_group)
        else:
            return apply_shard_shapes(x, dim, shard_shapes_dim)

    def _apply_truncation(self, x, grid_shard_shapes=None, model_comm_group=None):
        if self.A_down is not None or self.A_up is not None:
            if grid_shard_shapes is not None:
                shard_shapes = self._get_shard_shapes(x, 0, grid_shard_shapes, model_comm_group)
                # grid-sharded input: reshard to channel-shards to apply truncation
                x = shard_channels(x, shard_shapes, model_comm_group)  # we get the full sequence here

            # these can't be registered as buffers because ddp does not like to broadcast sparse tensors
            # hence we check that they are on the correct device ; copy should only happen in the first forward run
            if self.A_down is not None:
                self.A_down = self.A_down.to(x.device)
                x = self._truncate_fields(x, self.A_down)  # to coarse resolution
            if self.A_up is not None:
                self.A_up = self.A_up.to(x.device)
                x = self._truncate_fields(x, self.A_up)  # back to high resolution

            if grid_shard_shapes is not None:
                # back to grid-sharding as before
                x = gather_channels(x, shard_shapes, model_comm_group)

        return x
