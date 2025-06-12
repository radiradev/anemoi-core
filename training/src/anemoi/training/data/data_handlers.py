import copy
import logging
import random
from datetime import timedelta
from enum import Enum
from functools import cached_property
from typing import Any
from typing import Literal

import einops
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from anemoi.utils.config import DotDict
from omegaconf import ListConfig
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



class AbstractDataHandler: # not so abstract -> rename it
    def __init__(self, *args, **kwargs):
        pass

    # TODO: add
    def variables(self) -> list[str]:
        raise NotImplementedError("do we need to implement this?")
    
    @property
    def _dataset(self):
        raise NotImplementedError("Subclasses must implement _dataset property")

    @property
    def name_to_index(self):
        raise NotImplementedError("Subclasses must implement name_to_index property")

    @property
    def statistics(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement statistics property")

    @property
    def frequency(self) -> timedelta:
        raise NotImplementedError("Subclasses must implement frequency property")

    @property
    def start_date(self):
        raise NotImplementedError("Subclasses must implement start_date property")

    @property
    def end_date(self):
        raise NotImplementedError("Subclasses must implement end_date property")

    @property
    def groups(self) -> tuple[str]:
        raise NotImplementedError("Subclasses must implement groups property")

    def __getitem__(self, key, *args, **kwargs) :
        raise NotImplementedError("Subclasses must implement __getitem__ method")

    def processors(self) -> list:
        raise NotImplementedError("Subclasses must implement processors method")

    def select(self, select: list[str] | None =None):
        if select is None:
            return self
        return SelectedDataHandler(self, select=select)


class NongroupedDataHandler(AbstractDataHandler):
    def __init__(self, dataset, processors = None, default_group: str = "data"):
        self._dataset_config = dataset
        self._processors = processors
        self.name = default_group

    @property
    def _dataset(self):
        return open_dataset(self._dataset_config)
    
    @property
    def name_to_index(self) -> dict[str, dict]:
        return {self.name: self._dataset.name_to_index}

    @property
    def groups(self) -> tuple[str]:
        return (self.name, )

    @property
    def statistics(self) -> dict[str, torch.Tensor]:
        return {self.name: self._dataset.statistics}

    def __getitem__(self, *args, **kwargs) -> dict[GroupName, np.ndarray]:
        return {self.name: self._dataset.__getitem__(*args, **kwargs)}


class GroupedDataHandler(AbstractDataHandler):
    def __init__(self, dataset, processors = None):
        self._dataset_config = dataset
        self._processors = processors

    @property
    def _dataset(self):
        return open_dataset(self._dataset_config)
    
    @property
    def name_to_index(self) -> dict[str, dict]:
        return self._dataset.name_to_index

    @property
    def groups(self) -> tuple[str]:
        return self._dataset.groups

    @property
    def statistics(self) -> dict[str, torch.Tensor]:
        return self._dataset.statistics

    def __getitem__(self, *args, **kwargs) -> dict[GroupName, np.ndarray]:
        return self._dataset.__getitem__(*args, **kwargs)


class MultiDataHandler(AbstractDataHandler):
    def __init__(self, data_handlers: list[AbstractDataHandler]):
        self._data_handlers = data_handlers

    @property
    def name_to_index(self) -> dict[GroupName, dict[str, int]]:
        dic = {}
        for data_handler in self._data_handlers:
            new = data_handler.name_to_index
            if set(dic.keys()).intersection(new.keys()):
                raise ValueError(f"Duplicate keys found in data handlers: {set(dic.keys()).intersection(new.keys())}")
            dic.update(new)
        return dic

    @property
    def statistics(self) -> dict[GroupName, dict[str, torch.Tensor]]:
        stats = {}
        for data_handler in self._data_handlers:
            stats.update(data_handler.statistics)
        return stats

    @property
    def frequency(self) -> timedelta:
        if not self._data_handlers:
            raise ValueError("No data handlers available to determine frequency.")
        frequencies = [dh.frequency for dh in self._data_handlers]
        if len(set(frequencies)) > 1:
            raise ValueError("Data handlers have different frequencies.")
        return frequencies[0]

    @property
    def start_date(self):
        # do we need this?
        if not self._data_handlers:
            raise ValueError("No data handlers available to determine start date.")
        start_dates = [dh.start_date for dh in self._data_handlers]
        if len(set(start_dates)) > 1:
            raise ValueError("Data handlers have different start dates.")
        return start_dates[0]

    @property
    def end_date(self):
        # do we need this?
        if not self._data_handlers:
            raise ValueError("No data handlers available to determine end date.")
        end_dates = [dh.end_date for dh in self._data_handlers]
        if len(set(end_dates)) > 1:
            raise ValueError("Data handlers have different end dates.")
        return end_dates[0]

    def __getitem__(self, key: str, *args, **kwargs):
        if key not in self.groups:
            raise KeyError(f"Group {key} not found in data handlers. Available groups: {self.groups}")

        item = None
        for dh in self._data_handlers:
            if key not in dh.groups:
                continue
            return dh.__getitem__(*args, **kwargs)[key]
        raise KeyError(f"Group {key} not found in any data handler. Available groups: {self.groups}")

    def processors(self) -> dict[str, list[BasePreprocessor]]:
        processors = {}
        for data_handler in self._data_handlers:
            for name, processor in data_handler.processors():
                if name not in processors:
                    processors[name] = []
                processors[name].append(processor)
        return processors

    def check_no_overlap(self, other: "DataHandlers") -> None:
        print('❌ TODO check for overlap')
        return
        for key, dh in self.items():
            if other[key].start_date < dh.end_date:
                raise ValueError
            # TODO: What do we want to check ???
            # no_overlap vs is_completely_before

    @cached_property
    def groups(self) -> list[GroupName]:
        """Return the list of group names."""
        groups = set()
        for dh in self._data_handlers:
            new = dh.groups
            if set(new).intersection(groups):
                raise ValueError(f"Duplicate group names found: {set(new).intersection(groups)}")
            groups.update(new)
        return tuple(groups)


def set_group(config, group: str):
    config = config.copy()
    config["dataset"] = {"dataset": config["dataset"], "set_group": group}
    return config


def data_handler_factory(config, top_level: bool = False) -> AbstractDataHandler:
    if isinstance(config, (dict, DotDict, DictConfig)):
        if top_level:
            new_config = [set_group(v, k) for k, v in config.items()]
            return data_handler_factory(new_config)

        assert "dataset" in config, f"Expected 'dataset' key in data handler.\nconfig,{config}\ntype: {type(config)}"
        assert "_target_" in config, config
        return instantiate(config, _recursive_=False)

    if isinstance(config, (list, tuple, ListConfig)):
        return MultiDataHandler(data_handler_factory(c) for c in config)
    
    raise ValueError(f"Data handler config of type {type(config)} is not supported. It should be a list or a dict.")


class ForwardDataHandler(AbstractDataHandler):
    def __init__(self,forward):
        self._forward = forward
    def name_to_index(self) -> dict[str, int]:
        return self._forward.name_to_index
    def statistics(self) -> dict[str, torch.Tensor]:
        return self._forward.statistics
    def frequency(self) -> timedelta:
        return self._forward.frequency
    def start_date(self):
        return self._forward.start_date
    def end_date(self):
        return self._forward.end_date
    def __getitem__(self, *args, **kwargs):
        return self._forward.__getitem__(*args, **kwargs)
    def processors(self) -> list[BasePreprocessor]:
        return self._forward.processors()
    def groups(self) -> tuple[str]:
        return self._forward.groups()

class SelectedDataHandler(ForwardDataHandler):
    def __init__(self, dh: AbstractDataHandler, select: list[str] = None):
        super().__init__(dh)
        selection = self._parse_select(select)
        assert isinstance(selection, dict), f"Selection must be a dict of list, not {type(selection)}"
        assert isinstance(selection[selection.keys()[0]], list), f"Selection values must be lists, not {type(selection[selection.keys()[0]])}"
        self._selection = selection
    def _parse_select(self, select):
        raise NotImplementedError("TODO implement _parse_select")
        return selection

    def _select(self, dict_of_dicts: dict[str: dict[str, Any]]) -> dict[str, dict[str, Any]]:
        d = copy.deepcopy(dict_of_dicts)
        # TODO select in dict of dict
        return d
    def name_to_index(self) -> dict[str, int]:
        return self._select(self._forward.name_to_index)
    def statistics(self) -> dict[str, torch.Tensor]:
        return self._select(self._forward.statistics)
    def processors(self):
        return self._select(self._forward.processors())
    def __getitem__(self, *args, **kwargs):
        """Get item from the forward data handler, applying selection if specified."""
        item = self._forward.__getitem__(*args, **kwargs)
        if self._selection is None:
            return item
        return {k: v for k, v in item.items() if k in self._selection}
    def groups(self) -> tuple[str]:
        all_groups = self._forward.groups()
        if self._selection is None:
            return all_groups
        return [g for g in all_groups if g in self._selection]



class RecordProvider:
    def __init__(self, kwargs: dict, datahandlers: "AbstractDataHandler") -> None:
        self._steps = {}
        _data_handlers = []

        for key, provider_config in kwargs.items():
            if key not in datahandlers.groups:
                raise ValueError(f"Unknown group: {key}, should be one of {list(datahandlers.groups)}")
            selection = dict(key=provider_config["variables"])
            dh = datahandlers.select(selection)

            steps = provider_config.get("steps",0)
            if isinstance(steps, int):
                steps = [steps]
            assert isinstance(steps, (list, tuple)), f"Steps must be a list or tuple, not {type(steps)}"
            self._steps[key] =steps

            _data_handlers.append(dh)
            #print("record provider", key, _data_handlers.variables)

        self._data_handlers = MultiDataHandler(_data_handlers)

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

    def get_sample_coverage(self) -> dict[GroupName, tuple[np.datetime64, np.datetime64, timedelta]]:
        # Dot we need this, is this just to check the config?
        coverage = {}
        for dh_key in self.group_names:
            coverage[dh_key] = (
                self._data_handlers[dh_key].start_date,
                self._data_handlers[dh_key].end_date,
                self._data_handlers[dh_key].frequency,
            )
        return coverage

    def get_steps(self, *args, **kwargs):
        assert False, "use .steps property instead"

    @property
    def steps(self) -> dict[GroupName, list[int]]:
        return self._steps

    def __getitem__(self, i: int) -> dict[GroupName, list[torch.Tensor]]:
        # Get the i-th record for each group, where i is the time index.
        # i is the time index, which is the same for all groups.
        # The data handlers are expected to return the data in the shape:
        # [C-]G-S-BT
        records = {}
        for group, steps in self.steps.items():
            x_s = []
            for step in steps:
                x = self.data_handlers[group, i + step]
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
