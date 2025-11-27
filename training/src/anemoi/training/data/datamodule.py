# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from functools import cached_property

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.data.dataset import NativeGridDataset
from anemoi.training.utils.worker_init import worker_init_func

LOGGER = logging.getLogger(__name__)


class AnemoiDatasetsDataModule(pl.LightningDataModule):
    """Anemoi Datasets data module for PyTorch Lightning."""

    def __init__(
        self,
        ds_train: NativeGridDataset,
        ds_valid: NativeGridDataset,
        ds_test: NativeGridDataset,
        batch_size_train: int,
        batch_size_valid: int,
        batch_size_test: int,
        num_workers_train: int,
        num_workers_valid: int,
        num_workers_test: int,
        pin_memory: bool,
        prefetch_factor: int,
    ) -> None:
        """Initialize Anemoi Datasets data module."""
        super().__init__()
        self.ds_train = ds_train
        self.ds_valid = ds_valid
        self.ds_test = ds_test
        self.batch_size_train = batch_size_train
        self.batch_size_valid = batch_size_valid
        self.batch_size_test = batch_size_test
        self.num_workers_train = num_workers_train
        self.num_workers_valid = num_workers_valid
        self.num_workers_test = num_workers_test
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    @cached_property
    def statistics(self) -> dict:
        return self.ds_train.statistics

    @cached_property
    def statistics_tendencies(self) -> dict:
        return self.ds_train.statistics_tendencies

    @cached_property
    def metadata(self) -> dict:
        return self.ds_train.metadata

    @cached_property
    def supporting_arrays(self) -> dict:
        return self.ds_train.supporting_arrays

    @cached_property
    def data_indices(self) -> IndexCollection:
        return IndexCollection(self.ds_train.config, self.ds_train.name_to_index)

    def _get_dataloader(self, ds: NativeGridDataset, stage: str) -> DataLoader:
        batch_size = getattr(self, f"batch_size_{stage}")
        num_workers = getattr(self, f"num_workers_{stage}")
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_func,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_train, "training")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_valid, "validation")

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_test, "test")
