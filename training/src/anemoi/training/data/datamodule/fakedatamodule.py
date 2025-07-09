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
from typing import TYPE_CHECKING

from anemoi.training.data.dataset import FakeNativeGridDataset

from .singledatamodule import AnemoiDatasetsDataModule

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable


class AnemoiFakeDatasetsDataModule(AnemoiDatasetsDataModule):
    """Anemoi Ensemble data module for PyTorch Lightning."""

    def _get_dataset(
        self,
        data_reader: Callable,
        shuffle: bool = True,
        val_rollout: int = 1,
        label: str = "generic",
    ) -> FakeNativeGridDataset:

        data_reader = self.add_trajectory_ids(data_reader)  # NOTE: Functionality to be moved to anemoi datasets

        return FakeNativeGridDataset(
            data_reader=data_reader,
            relative_date_indices=self.relative_date_indices(val_rollout),
            timestep=self.config.data.timestep,
            shuffle=shuffle,
            grid_indices=self.grid_indices,
            label=label,
        )
