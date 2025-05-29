import logging
import random
from datetime import timedelta
from enum import Enum
from typing import Literal

import einops
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import IterableDataset

from anemoi.datasets.data import open_dataset
from anemoi.models.preprocessing import BasePreprocessor
from anemoi.training.data.sampler import AnemoiSampler
from anemoi.training.data.utils import GroupName
from anemoi.training.data.utils import RecordSpec
from anemoi.training.data.utils import SampleSpec
from anemoi.training.data.utils import SourceSpec
LOGGER = logging.getLogger(__name__)


class Stage(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"



class DataHandlers(dict):
    def __init__(self, config: dict[str, dict]):
        for name, data_hanlder in config.items():
            if isinstance(data_hanlder, BaseDataHandler):
                self[name] = data_hanlder
            else:
                self[name] = DataHandler(**data_hanlder)

    @property
    def name_to_index(self) -> dict:
        return {key: data_handler.name_to_index for key, data_handler in self.items()}

    def processors(self) -> dict[str, list[BasePreprocessor]]:
        return {dh_name: data_handler.processors() for dh_name, data_handler in self.items()}

    def check_no_overlap(self, other: "DataHandlers") -> None:
        for key, dh in self.items():
            if other[key].start_date < dh.end_date:
                raise ValueError

            # TODO: What do we want to check ???
            # no_overlap vs is_completely_before


class BaseDataHandler:
    def __init__(self, dataset: str | dict, processors: DictConfig | None = None):
        self._dataset = open_dataset(dataset)
        self.variables = None
        self._processors = processors

    @property
    def name_to_index(self):
        return self._dataset.name_to_index

    def processors(self) -> list:
        return [[name, instantiate(processor, dataset=self._dataset)] for name, processor in self._processors.items()]

    @property
    def frequency(self) -> timedelta:
        return self._dataset.frequency

    @property
    def start_date(self):
        return self._dataset.start_date

    @property
    def end_date(self):
        return self._dataset.end_date

    def __getitem__(self, *args, **kwargs):
        return self._dataset.__getitem__(*args, **kwargs)   


class DataHandler(BaseDataHandler):
    pass


class SelectedDataHandler(BaseDataHandler):
    def __init__(self, dh, select=None):
        self._dataset = open_dataset(dh._dataset, select=select)
        self.variables = select
        self._processors = dh._processors


def select(data_handler, select=None):
    if select is None:
        return data_handler
    return SelectedDataHandler(data_handler, select=select)


class RecordProvider:
    def __init__(self, kwargs: dict, datahandlers: "DataHandlers") -> None:
        self._steps = {}
        _data_handlers = {}

        for key, provider_config in kwargs.items():
            if key not in datahandlers:
                raise ValueError(f"Unknown data handler: {key}, should be one of {list(datahandlers.keys())}")

            variables = list(provider_config["variables"])
            assert isinstance(variables, (list, tuple)), f"variables must be a list or tuple, not {type(variables)}"
            self._steps[key] = (
                list(provider_config["steps"])
                if isinstance(provider_config["steps"], int)
                else provider_config["steps"]
            )
            _data_handlers[key] = select(datahandlers[key], variables)

        self.data_handlers = DataHandlers(_data_handlers)

    @property
    def keys(self) -> list[GroupName]:
        return list(self.data_handlers.keys())

    @property
    def spec(self) -> RecordSpec:
        spec = {}
        for name, data_handler in self.data_handlers.items():
            spec[name] = SourceSpec(data_handler.variables, self._steps[name])
        return RecordSpec(spec)

    def processors(self) -> list[BasePreprocessor]:
        return self.data_handlers.processors()

    def get_sample_coverage(self) -> dict[GroupName, tuple[np.datetime64, np.datetime64, timedelta]]:
        coverage = {}
        for dh_key in self.keys:
            coverage[dh_key] = (
                self.data_handlers[dh_key].start_date,
                self.data_handlers[dh_key].end_date,
                self.data_handlers[dh_key].frequency,
            )
        return coverage

    def get_steps(self, i: int) -> dict[GroupName, list[int]]:
        steps = {}
        for dh_key in self.keys:
            steps[dh_key] = [i + l for l in self._steps[dh_key]]
        return steps

    def __getitem__(self, i: int) -> dict[GroupName, list[torch.Tensor]]:
        records = {}
        for group_name, dh_steps in self.get_steps(i).items():
            records[group_name] = self.data_handlers[group_name][dh_steps, :]

        return records


class SampleProvider:
    def __init__(
        self,
        provider_config: DictConfig,
        data_handlers: DataHandlers,
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

    def __getitem__(self, i: int) -> dict[Literal["input", "target"], "Thing"]:
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

        self.sampler = AnemoiSampler(sample_provider.input, sample_provider.target, **sampler_config)

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
