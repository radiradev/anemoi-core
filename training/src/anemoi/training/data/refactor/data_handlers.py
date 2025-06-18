import copy
import logging
from datetime import timedelta
from functools import cached_property
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from anemoi.utils.config import DotDict
from omegaconf import ListConfig

from anemoi.datasets.data import open_dataset
from anemoi.models.preprocessing import BasePreprocessor
from anemoi.training.data.utils import GroupName

LOGGER = logging.getLogger(__name__)


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

    def select(self, select: dict[str, list[str]] | None =None):
        if select is None:
            return self
        assert len(self.groups) == 1, self.groups
        if self.groups[0] not in select:
            # What should select=None return? all variables or none?
            return self

        return SelectedDataHandler(self, select=select[self.groups[0]])


class ForwardDataHandler(AbstractDataHandler):
    def __init__(self, forward):
        self._forward = forward

    @property
    def name_to_index(self) -> dict[str, int]:
        return self._forward.name_to_index

    @property
    def statistics(self) -> dict[str, torch.Tensor]:
        return self._forward.statistics

    @property
    def frequency(self) -> timedelta:
        return self._forward.frequency

    @property
    def start_date(self):
        return self._forward.start_date

    @property
    def end_date(self):
        return self._forward.end_date

    def __getitem__(self, *args, **kwargs):
        return self._forward.__getitem__(*args, **kwargs)

    def processors(self) -> list[BasePreprocessor]:
        return self._forward.processors()
    
    @property
    def groups(self) -> tuple[str]:
        return self._forward.groups


class AnemoiDataHandler(ForwardDataHandler):
    def __init__(self, dataset, processors = None):
        super().__init__(open_dataset(dataset))
        self._dataset_config = dataset
        self._processors = processors

    @property
    def frequency(self) -> np.timedelta64:
        return np.timedelta64(super().frequency)

    @property
    def start_date(self) -> np.datetime64:
        return np.datetime64(super().start_date)

    @property
    def end_date(self) -> np.datetime64:
        return np.datetime64(super().end_date)


class MultiDataHandler(AbstractDataHandler):
    def __init__(self, data_handlers: list[AbstractDataHandler]):
        self._data_handlers = list(data_handlers)

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
            LOGGER.warning("Data handlers have different start dates.")
        return min(start_dates)

    @property
    def end_date(self):
        # do we need this?
        if not self._data_handlers:
            raise ValueError("No data handlers available to determine end date.")
        end_dates = [dh.end_date for dh in self._data_handlers]
        if len(set(end_dates)) > 1:
            LOGGER.warning("Data handlers have different end dates.")
        return max(end_dates)

    def __getitem__(self, args, **kwargs):
        key = args[0]
        if key not in self.groups:
            raise KeyError(f"Group {key} not found in data handlers. Available groups: {self.groups}")

        item = None
        for dh in self._data_handlers:
            if key not in dh.groups:
                continue
            return dh.__getitem__(*args[1:], **kwargs)[key]
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

    def select(self, select: dict[str, list[str]] | None = None):
        self._data_handlers = [dh.select(select) for dh in self._data_handlers]


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
        return AnemoiDataHandler(**config)

    if isinstance(config, (list, tuple, ListConfig)):
        return MultiDataHandler(data_handler_factory(c) for c in config)
    
    raise ValueError(f"Data handler config of type {type(config)} is not supported. It should be a list or a dict.")


class SelectedDataHandler(ForwardDataHandler):
    def __init__(self, dh: AbstractDataHandler, select: list[str] = None):
        super().__init__(dh)
        assert isinstance(select, (list, ListConfig)), f"Selection values must be lists, not {type(select)}"
        self._selection = select

    def _select(self, dict_of_dicts: dict[str: dict[str, Any]]) -> dict[str, dict[str, Any]]:
        d = copy.deepcopy(dict_of_dicts)
        # TODO select in dict of dict
        return d

    @property
    def name_to_index(self) -> dict[str, int]:
        return self._select(self._forward.name_to_index)

    @property
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
