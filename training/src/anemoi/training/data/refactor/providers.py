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
