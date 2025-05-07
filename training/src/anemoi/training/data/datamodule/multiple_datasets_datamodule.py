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
from anemoi.training.data.utils import SamplerProviderName
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
        self.sample_providers = {
            key: SampleProvider(provider, dhs) for key, provider in config.model.sample_providers.items()
        }

        # Create samplers
        self.train_sampler = AnemoiSampler(**config.dataloader.sampler.training)
        self.val_sampler = AnemoiSampler(**config.dataloader.sampler.validation)
        self.test_sampler = AnemoiSampler(**config.dataloader.sampler.test)
        
        self.train_sampler.set_valid_indices(self.sample_providers)
        self.val_sampler.set_valid_indices(self.sample_providers)
        self.test_sampler.set_valid_indices(self.sample_providers)

        dl_keys_to_ignore = ["sampler", "read_group_size", "grid_indices", "limit_batches"]
        self.train_dataloader_config = get_dataloader_config(config.dataloader, "training", keys_to_ignore=dl_keys_to_ignore)
        self.val_dataloader_config = get_dataloader_config(config.dataloader, "validation", keys_to_ignore=dl_keys_to_ignore)
        self.test_dataloader_config = get_dataloader_config(config.dataloader, "test", keys_to_ignore=dl_keys_to_ignore)

        # data_handlers[stage.TRAINING].check_no_overlap(data_handlers[stage.VALIDATION])
        # data_handlers[stage.TRAINING].check_no_overlap(data_handlers[stage.TEST])
        # data_handlers[stage.VALIDATION].check_no_overlap(data_handlers[stage.TEST])

    @cached_property
    def data_indices(self) -> IndexCollection:
        return IndexCollection(self.config, self.ds_train.name_to_index)

    def _get_dataloaders(
        self,
        sample_providers: dict[SamplerProviderName, SampleProvider],
        sampler: AnemoiSampler,
        **kwargs: dict,
    ) -> dict[SamplerProviderName, DataLoader]:
        data_loaders = {
            name: DataLoader(
                NativeGridMultDataset(sample_provider, sampler),
                # worker initializer
                worker_init_fn=worker_init_func,
                **kwargs
            )
            for name, sample_provider in sample_providers.items()
        }
        return data_loaders

    def train_dataloader(self) -> dict[str, DataLoader]:
        return self._get_dataloaders(self.sample_providers, self.train_sampler, **self.train_dataloader_config)

    def val_dataloader(self) -> dict[str, DataLoader]:
        return self._get_dataloaders(self.sample_providers, self.val_sampler, **self.val_dataloader_config)

    def test_dataloader(self) -> dict[str, DataLoader]:
        return self._get_dataloaders(self.sample_providers, self.test_sampler, **self.test_dataloader_config)
