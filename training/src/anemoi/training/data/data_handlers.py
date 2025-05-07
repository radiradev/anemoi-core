import logging
import random
from datetime import timedelta
from enum import Enum

import einops
import numpy as np
import torch
from torch.utils.data import IterableDataset

from anemoi.datasets.data import open_dataset
from anemoi.training.data.utils import DataHandlerName

LOGGER = logging.getLogger(__name__)


class Stage(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"


class DataHandlers(dict):
    def __init__(self, config: dict[str, dict]):
        for name, kwargs in config.items():
            self[name] = DataHandler(**kwargs)

    def check_no_overlap(self, other: "DataHandlers") -> None:
        for key, dh in self.items():
            if other[key].start_date < dh.end_date:
                raise ValueError

            # TODO: What do we want to check ???
            # no_overlap vs is_completely_before


class BaseDataHandler:
    def __init__(self, dataset: str | dict, processors: list | None = None):
        self._dataset = open_dataset(dataset)
        self.variables = None
        self.processors = processors

    @property
    def frequency(self) -> timedelta:
        return self._dataset.frequency

    @property
    def start_date(self):
        return self._dataset.start_date

    @property
    def end_date(self):
        return self._dataset.end_date


class DataHandler(BaseDataHandler):
    pass


class SelectedDataHandler(BaseDataHandler):
    def __init__(self, dh, select=None):
        self._dataset = open_dataset(dh._dataset, select=select)
        self.processors = dh.processors


def select(data_handler, select=None):
    if select is None:
        return data_handler
    return SelectedDataHandler(data_handler, select=select)


class SampleProvider:
    def __init__(self, kwargs: dict, datahandlers: "DataHandlers") -> None:
        self._variables = {}
        self._data_handlers = {}
        self._steps = {}

        for key, provider_config in kwargs.items():
            if key not in datahandlers:
                raise ValueError(f"Unknown data handler: {key}, should be one of {list(datahandlers.keys())}")

            variables = list(provider_config["variables"])
            assert isinstance(variables, (list, tuple)), f"variables must be a list or tuple, not {type(variables)}"
            self._variables[key] = variables
            self._data_handlers[key] = select(datahandlers[key], variables)
            self._steps[key] = (
                list(provider_config["steps"])
                if isinstance(provider_config["steps"], int)
                else provider_config["steps"]
            )

    @property
    def keys(self) -> list[DataHandlerName]:
        return list(self._data_handlers.keys())

    def get_sample_coverage(self) -> dict[DataHandlerName, tuple[np.datetime64, np.datetime64, timedelta]]:
        coverage = {}
        for dh_key in self.keys:
            coverage[dh_key] = (
                self._data_handlers[dh_key].start_date,
                self._data_handlers[dh_key].end_date,
                self._data_handlers[dh_key].frequency,
            )
        return coverage

    def get_steps(self, i: int) -> dict[DataHandlerName, int | list[int]]:
        steps = {}
        for dh_key in self.keys:
            steps[dh_key] = [i + l for l in self._steps[dh_key]]
        return steps

    def __getitem__(self, i: int) -> dict[DataHandlerName, torch.Tensor]:
        sample = {}
        for dh_name, dh_steps in self.get_steps(i).items():
            x = self._data_handlers[dh_name]._dataset[dh_steps, :, :, :]
            x = einops.rearrange(x, "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")
            self.ensemble_dim = 1
            sample[dh_name] = torch.from_numpy(x)
        return sample


class NativeGridMultDataset(IterableDataset):
    """Iterable dataset for AnemoI data on the arbitrary grids."""

    def __init__(
        self,
        data_reader: SampleProvider,
        sampler: "BaseAnemoiSampler",
        shuffle: bool = True,
    ) -> None:
        self.data = data_reader
        self.sampler = sampler
        self.shuffle = shuffle

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
        base_seed = 2025 #get_base_seed()

        torch.manual_seed(base_seed)
        random.seed(base_seed)
        self.rng = np.random.default_rng(seed=base_seed)

    def __iter__(self) -> dict[DataHandlerName, torch.Tensor]:
        """Return an iterator over the dataset.

        The datasets are retrieved by anemoi.datasets from anemoi datasets. This iterator yields
        chunked batches for DDP and sharded training.

        Currently it receives data with an ensemble dimension, which is discarded for
        now. (Until the code is "ensemble native".)
        """
        valid_indices = self.sampler.valid_time_indices

        for i in valid_indices:
            yield self.data[i]
