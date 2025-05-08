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
from anemoi.training.data.utils import get_dataloader_config
from anemoi.training.data.sampler import AnemoiSampler
from anemoi.training.data.utils import RecordProviderName
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.utils.worker_init import worker_init_func
from anemoi.utils.dates import frequency_to_seconds

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:

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

        # Create data handlers            
        dhs = DataHandlers(config.data.data_handlers)
        
        # Create Sampler provider
        self.sample_provider = SampleProvider(config.model, dhs)

        # Create datasets
        self.train_dataset = NativeGridMultDataset(self.sample_provider, sampler_config=config.dataloader.sampler.training)
        self.val_dataset = NativeGridMultDataset(self.sample_provider, sampler_config=config.dataloader.sampler.validation)

        dl_keys_to_ignore = ["sampler", "read_group_size", "grid_indices", "limit_batches"]
        self.train_dataloader_config = get_dataloader_config(config.dataloader, "training", keys_to_ignore=dl_keys_to_ignore)
        self.val_dataloader_config = get_dataloader_config(config.dataloader, "validation", keys_to_ignore=dl_keys_to_ignore)

        # data_handlers[stage.TRAINING].check_no_overlap(data_handlers[stage.VALIDATION])
        # data_handlers[stage.TRAINING].check_no_overlap(data_handlers[stage.TEST])
        # data_handlers[stage.VALIDATION].check_no_overlap(data_handlers[stage.TEST])

    @cached_property
    def data_indices(self) -> IndexCollection:
        return IndexCollection(self.config, self.ds_train.name_to_index)

    def train_dataloader(self) -> dict[str, DataLoader]:
        return DataLoader(self.train_dataset, worker_init_fn=worker_init_func, **self.train_dataloader_config)

    def val_dataloader(self) -> dict[str, DataLoader]:
        return DataLoader(self.val_dataset, worker_init_fn=worker_init_func, **self.val_dataloader_config)
