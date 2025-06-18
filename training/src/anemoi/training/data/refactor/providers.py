import logging
import random
from typing import Literal

import einops
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import IterableDataset

from anemoi.models.preprocessing import BasePreprocessor
from anemoi.training.data.refactor.data_handlers import AbstractDataHandler
from anemoi.training.data.refactor.sampler import AnemoiSampler
from anemoi.training.data.utils import GroupName
from anemoi.training.data.utils import RecordSpec
from anemoi.training.data.utils import SampleSpec
from anemoi.training.data.utils import SourceSpec

LOGGER = logging.getLogger(__name__)


def remove_config_level(config, key: str, default = None):
    return {k: v.get(key, default) for k, v in config.items()}


class RecordProvider:
    def __init__(self, kwargs: dict, datahandlers: "AbstractDataHandler") -> None:
        self._data_handlers = datahandlers
        
        steps_config = remove_config_level(kwargs, "steps", default=0)
        vars_config = remove_config_level(kwargs, "variables", default=[])
        
        self._steps = steps_config
        self._data_handlers.select(vars_config)

    @property
    def group_names(self) -> list[GroupName]:
        return list(self._data_handlers.groups)

    @property
    def name_to_index(self) -> dict[GroupName, dict[str, int]]:
        return self._data_handlers.name_to_index

    @property
    def statistics(self) -> dict[GroupName, dict[str, torch.Tensor]]:
        return self._data_handlers.statistics

    @property
    def spec(self) -> RecordSpec:
        # Do we need this ?
        spec = {}
        for name, data_handler in self._data_handlers.items():
            spec[name] = SourceSpec(data_handler.variables, self._steps[name])
        return RecordSpec(spec)

    def processors(self) -> list[BasePreprocessor]:
        return self._data_handlers.processors()

    def get_valid_times(self, time_values: np.ndarray) -> np.ndarray:
        valid_times = np.full(len(time_values), True)

        for dh in self._data_handlers._data_handlers:
            start_date, end_date, freq_td = dh.start_date, dh.end_date, dh.frequency

            assert len(dh.groups) == 1, dh.groups
            steps = self._steps[dh.groups[0]] * freq_td
            min_valid_time = start_date - min(steps)
            max_valid_time = end_date - max(steps)
            is_within_range = (time_values >= min_valid_time) & (time_values <= max_valid_time)
            # TODO: Implement functionality for missing values
            # TODO: Implement "required" tag for dh
            valid_times &= is_within_range

        return valid_times

    def __getitem__(self, i: int) -> dict[GroupName, list[torch.Tensor]]:
        # Get the i-th record for each group, where i is the time index.
        # i is the time index, which is the same for all groups.
        # The data handlers are expected to return the data in the shape:
        # [C-]G-S-BT
        records = {}
        for group, steps in self._steps.items():
            x_s = []
            for step in steps:
                x = self._data_handlers[group, i + step]
                x = einops.rearrange(x, "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")
                self.ensemble_dim = 1
                x = torch.from_numpy(x)
                x_s.append(x)
            # TODO for obs: if the tensors x are not the same shape, we cannot stack them
            # TODO :check if torch.stack is the right stacking (vstack, hstack, ...)
            x_s = torch.stack(x_s, dim=1)  # [C-]G-S-BT
            records[group] = x_s
        return records


class SampleProvider:
    def __init__(
        self,
        provider_config: DictConfig,
        data_handlers: AbstractDataHandler,
    ) -> None:
        self.input = RecordProvider(provider_config.input_provider, data_handlers)
        self.target = RecordProvider(provider_config.target_provider, data_handlers)

    def input_processors(self) -> list[BasePreprocessor]:
        return self.input.processors()

    def target_processors(self) -> list[BasePreprocessor]:
        return self.target.processors()

    @property
    def spec(self) -> SampleSpec:
        return SampleSpec({"input": self.input.spec, "target": self.target.spec})

    def __getitem__(self, i: int) -> dict[Literal["input", "target"], dict[GroupName, dict[str, torch.Tensor]]]:
        return {"input": self.input[i], "target": self.target[i]}


class NativeGridMultDataset(IterableDataset):
    """Iterable dataset for AnemoI data on the arbitrary grids."""

    def __init__(
        self,
        sample_provider: SampleProvider,
        sampler_config: DictConfig,
        shuffle: bool = True,
    ) -> None:
        self.sample_provider = sample_provider
        self.shuffle = shuffle

        self.sampler = AnemoiSampler(**sampler_config)
        self.sampler.set_valid_indices(sample_provider.input, sample_provider.target)

        # lazy init
        self.n_samples_per_epoch_total: int = 0
        self.n_samples_per_epoch_per_worker: int = 0

        # lazy init model and reader group info, will be set by the DDPGroupStrategy:
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1
        self.model_comm_group_id = 0
        self.global_rank = 0

        self.reader_group_rank = 0
        self.reader_group_size = 1

        self.sample_comm_num_groups = 1  # groups that work on the same sample / batch
        self.sample_comm_group_id = 0

        # additional state vars (lazy init)
        self.n_samples_per_worker = 0
        self.chunk_index_range: np.ndarray | None = None
        self.shuffle = shuffle

        # Data dimensions
        self.ensemble_dim: int = 2
        self.ensemble_size = 1

    def set_comm_group_info(
        self,
        global_rank: int,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        """Set model and reader communication group information (called by DDPGroupStrategy).

        Parameters
        ----------
        global_rank : int
            Global rank
        model_comm_group_id : int
            Model communication group ID
        model_comm_group_rank : int
            Model communication group rank
        model_comm_num_groups : int
            Number of model communication groups
        reader_group_rank : int
            Reader group rank
        reader_group_size : int
            Reader group size
        """
        self.global_rank = global_rank
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

        self.sample_comm_group_id = model_comm_group_id
        self.sample_comm_num_groups = model_comm_num_groups

        assert self.reader_group_size >= 1, "reader_group_size must be positive"

        LOGGER.debug(
            "NativeGridDataset.set_group_info(): global_rank %d, model_comm_group_id %d, "
            "model_comm_group_rank %d, model_comm_num_groups %d, reader_group_rank %d",
            global_rank,
            model_comm_group_id,
            model_comm_group_rank,
            model_comm_num_groups,
            reader_group_rank,
        )

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Called by worker_init_func on each copy of dataset.

        This initialises after the worker process has been spawned.

        Parameters
        ----------
        n_workers : int
            Number of workers
        worker_id : int
            Worker ID

        """
        self.worker_id = worker_id
        base_seed = 2025  # get_base_seed()

        torch.manual_seed(base_seed)
        random.seed(base_seed)
        self.rng = np.random.default_rng(seed=base_seed)

    def __iter__(self) -> dict[GroupName, dict[str, torch.Tensor]]:
        """Return an iterator over the dataset.

        The datasets are retrieved by anemoi.datasets from anemoi datasets. This iterator yields
        chunked batches for DDP and sharded training.

        Currently it receives data with an ensemble dimension, which is discarded for
        now. (Until the code is "ensemble native".)
        """
        valid_indices = self.sampler.valid_time_indices

        for i in valid_indices:
            yield self.sample_provider[i]
