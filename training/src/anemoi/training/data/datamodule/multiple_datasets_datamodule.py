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
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import pytorch_lightning as pl
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.data.data_handlers import DataHandlers
from anemoi.training.data.data_handlers import NativeGridMultDataset
from anemoi.training.data.data_handlers import SampleProvider
from anemoi.training.data.data_handlers import Stage
from anemoi.training.data.utils import specify_datahandler_config
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.utils.worker_init import worker_init_func
from anemoi.utils.dates import frequency_to_seconds

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch_geometric.data import HeteroData

    from anemoi.training.data.grid_indices import BaseGridIndices
    from anemoi.training.schemas.base_schema import BaseSchema


class AnemoiMultipleDatasetsDataModule(pl.LightningDataModule):
    """Anemoi Datasets data module for PyTorch Lightning."""

    def __init__(self, config: BaseSchema, graph_data: HeteroData) -> None:
        """Initialize Anemoi Datasets data module.

        Parameters
        ----------
        config : BaseSchema
            Job configuration
        """
        super().__init__()

        self.config = config
        self.graph_data = graph_data

        # Create data handler for training, validation and testing.
        data_handlers = {}
        self.sample_providers = {}
        self.datasets = {}
        self.sampler = {}

        for stage in Stage:
            dh_configs = {
                k: specify_datahandler_config(v, f"{stage.value}_dataset") for k, v in config.data.data_handlers.items()
            }
            dhs = DataHandlers(dh_configs)
            data_handlers[stage] = dhs

            self.sample_providers[stage] = {
                key: SampleProvider(provider, dhs) for key, provider in config.model.sample_providers.items()
            }
            self.sampler[stage] = instantiate(config.dataloader.sampler)
            self.sampler[stage].set_valid_indices(self.sample_providers[stage])

        # data_handlers[stage.TRAINING].check_no_overlap(data_handlers[stage.VALIDATION])
        # data_handlers[stage.TRAINING].check_no_overlap(data_handlers[stage.TEST])
        # data_handlers[stage.VALIDATION].check_no_overlap(data_handlers[stage.TEST])

    @cached_property
    def statistics(self) -> dict:
        return self.ds_train.statistics

    @cached_property
    def metadata(self) -> dict:
        return self.ds_train.metadata

    @cached_property
    def supporting_arrays(self) -> dict:
        return self.ds_train.supporting_arrays | self.grid_indices.supporting_arrays

    @cached_property
    def data_indices(self) -> IndexCollection:
        return IndexCollection(self.config, self.ds_train.name_to_index)

    @cached_property
    def grid_indices(self) -> type[BaseGridIndices]:
        reader_group_size = self.config.dataloader.read_group_size

        grid_indices = instantiate(
            self.config.dataloader.grid_indices,
            reader_group_size=reader_group_size,
        )
        # grid_indices.setup(self.graph_data)
        return grid_indices

    @cached_property
    def timeincrement(self) -> int:
        """Determine the step size relative to the data frequency."""
        try:
            frequency = frequency_to_seconds(self.config.data.frequency)
        except ValueError as e:
            msg = f"Error in data frequency, {self.config.data.frequency}"
            raise ValueError(msg) from e

        try:
            timestep = frequency_to_seconds(self.config.data.timestep)
        except ValueError as e:
            msg = f"Error in timestep, {self.config.data.timestep}"
            raise ValueError(msg) from e

        assert timestep % frequency == 0, (
            f"Timestep ({self.config.data.timestep} == {timestep}) isn't a "
            f"multiple of data frequency ({self.config.data.frequency} == {frequency})."
        )

        LOGGER.info(
            "Timeincrement set to %s for data with frequency, %s, and timestep, %s",
            timestep // frequency,
            frequency,
            timestep,
        )
        return timestep // frequency

    def _get_dataloaders(self, stage: Stage) -> dict[str, DataLoader]:  # dict[str, dict[str, DataLoader]]
        # data_handlers : dict[str, DataHandlers]  ---> train/val/test | era5/cerra/...
        # sample_providers: dict[str, dict[str, SampleProviders]]  ---> train/val/test | input/output | era5/cerra/...
        # datasets:         dict[str, dict[str, NativeGridMultDatasets(inherits from IterableDataset)]]  ---> train/val/test | input/output | era5/cerra/...
        # dataloader: dict[str, torch.DataLoader]
        # datamodule: AnemoiMultipleDatasetsDataModule(pl.LightningDataModule)
        # samplers: dict[str, AnemoiSampler] for stage
        data_loaders = {
            name: DataLoader(
                NativeGridMultDataset(sample_provider, self.sampler[stage]),
                batch_size=self.config.dataloader.batch_size[stage.value],
                # number of worker processes
                num_workers=self.config.dataloader.num_workers[stage.value],
                # use of pinned memory can speed up CPU-to-GPU data transfers
                # see https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning
                pin_memory=self.config.dataloader.pin_memory,
                # worker initializer
                worker_init_fn=worker_init_func,
                # prefetch batches
                prefetch_factor=self.config.dataloader.prefetch_factor,
                persistent_workers=True,
            )
            for name, sample_provider in self.sample_providers[stage].items()
        }
        return data_loaders

    def train_dataloader(self) -> dict[str, DataLoader]:
        return self._get_dataloaders(Stage.TRAINING)

    def val_dataloader(self) -> dict[str, DataLoader]:
        return self._get_dataloaders(Stage.VALIDATION)

    def test_dataloader(self) -> dict[str, DataLoader]:
        return self._get_dataloaders(Stage.TEST)
