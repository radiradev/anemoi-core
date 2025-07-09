# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import torch

from anemoi.training.data.dataset import NativeGridDataset


class FakeNativeGridDataset(NativeGridDataset):

    def __iter__(self) -> torch.Tensor:
        timeincrement = self.relative_date_indices[1] - self.relative_date_indices[0]
        num_dates = ((self.relative_date_indices[-1] - self.relative_date_indices[0]) // timeincrement) + 1

        num_ensembles = self.data.shape[2]
        num_variables = self.data.shape[1]

        # Determine number of gridpoints for this shard
        grid_shard_indices = self.grid_indices.get_shard_indices(self.reader_group_rank)
        if isinstance(grid_shard_indices, slice):
            # Compute number of selected gridpoints from slice
            grid_slice = range(*grid_shard_indices.indices(self.data.shape[3]))
            num_gridpoints = len(grid_slice)
        else:
            num_gridpoints = len(grid_shard_indices)

        for _ in range(len(self.valid_date_indices)):
            fake_tensor = torch.randn((num_dates, num_ensembles, num_gridpoints, num_variables), device="cpu")
            yield fake_tensor
