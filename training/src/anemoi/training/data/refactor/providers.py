import logging
from typing import Literal

import einops
import numpy as np
import torch
from omegaconf import DictConfig

from anemoi.models.preprocessing import BasePreprocessor
from anemoi.training.data.refactor.data_handlers import AbstractDataHandler
from anemoi.training.data.utils import GroupName
from anemoi.training.data.utils import RecordSpec
from anemoi.training.data.utils import SampleSpec
from anemoi.training.data.utils import SourceSpec

LOGGER = logging.getLogger(__name__)


def remove_config_level(config, key: str, default=None):
    return {k: v.get(key, default) for k, v in config.items()}


def convert_to_timedelta(value: str) -> np.timedelta64:
    """Convert string to timedelta.
    
    Arguments
    ---------
    value : str
        The timedelta string. Options: 1D, 24h, 6h, 1h, 30m, 10m
    """
    time_res = value[-1]
    num = int(value[:-1])
    return np.timedelta64(num, time_res)


class RecordProvider:
    def __init__(self, kwargs: dict, datahandlers: "AbstractDataHandler") -> None:
        self._data_handlers = datahandlers

        steps_config = remove_config_level(kwargs, "steps", default=0)
        vars_config = remove_config_level(kwargs, "variables", default=[])
        self._required_group = remove_config_level(kwargs, "required", default=True)

        self._data_handlers.select(vars_config)
        self._time_steps = {k: [convert_to_timedelta(t) for t in steps] for k, steps in steps_config.items()}
        self._index_steps = {k: (v / self._data_handlers.frequency[k]).astype(int) for k, v in self._time_steps.items()}

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
            spec[name] = SourceSpec(data_handler.variables, self._time_steps[name])
        return RecordSpec(spec)

    def processors(self) -> list[BasePreprocessor]:
        return self._data_handlers.processors()

    def get_valid_times(self, time_values: np.ndarray) -> np.ndarray:
        valid_times = np.full(len(time_values), True)

        for group_name, steps in self._time_steps.items():
            if not self._required_group[group_name]:
                LOGGER.warning(f"{group_name} group is not required. Samples may not include this information.")
                continue

            start_date = self._data_handlers.start_date[group_name]
            end_date = self._data_handlers.end_date[group_name]
            freq_td = self._data_handlers.frequency[group_name]

            assert all(steps / freq_td % 1 == 0), f"steps, {steps}, must be multiple of freq, {freq_td}."
            min_valid_time = start_date - min(steps)
            max_valid_time = end_date - max(steps)
            is_within_range = (time_values >= min_valid_time) & (time_values <= max_valid_time)

            # TODO: Implement functionality for missing values
            valid_times &= is_within_range

        return valid_times

    def __getitem__(self, i: np.datetime64) -> dict[GroupName, list[torch.Tensor]]:
        # Get the i-th record for each group, where i is the time index.
        # i is the time index, which is the same for all groups.
        # The data handlers are expected to return the data in the shape:
        # [C-]G-S-BT
        records = {}
        for group, steps in self._index_steps.items():
            x_s = []
            dh_i = self._data_handlers.map_datetime_to_index(i)
            for step in steps:
                x = self._data_handlers.__getitem__(group, int(dh_i[group] + step))
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

    def __getitem__(self, i: np.datetime64) -> dict[Literal["input", "target"], dict[GroupName, dict[str, torch.Tensor]]]:
        return {"input": self.input[i], "target": self.target[i]}
