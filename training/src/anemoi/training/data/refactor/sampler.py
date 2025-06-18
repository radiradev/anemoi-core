# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABC
from abc import abstractmethod
from functools import cached_property

import numpy as np
import pandas as pd

from anemoi.training.data.utils import parse_date


class BaseAnemoiSampler(ABC):
    """Base AnemoiSampler class"""

    def __init__(self):
        self.valid_time_indices = None

    @abstractmethod
    def time_values(self) -> np.ndarray: ...

    def valid_time_values(self) -> np.ndarray:
        return self.time_values[self.valid_time_indices]

    @abstractmethod
    def set_valid_indices(self, *providers: "RecordProvider") -> list[int]: ...


class AnemoiSampler(BaseAnemoiSampler):
    """Sampler"""

    def __init__(self, frequency: str, start: str | int, end: str | int):
        self.frequency = frequency
        self.start = parse_date(start)
        self.end = parse_date(end)

    @cached_property
    def time_values(self) -> np.ndarray:
        """Time indices"""
        time_values = pd.date_range(start=self.start, end=self.end, freq=self.frequency)
        return np.array(time_values, dtype="datetime64[ns]")

    def set_valid_indices(self, *providers: "RecordProvider") -> list[int]:
        """Set the valid indices.

        This method set the valid refernce indices for sampling.
        """
        # TODO: Handle missing data
        valid_time_indices = np.full(len(self.time_values), True)

        for record_provider in providers:
            provider_valid_time_indices = record_provider.get_valid_times(self.time_values)
            valid_time_indices &= provider_valid_time_indices

        self.valid_time_indices = list(np.where(valid_time_indices)[0])
