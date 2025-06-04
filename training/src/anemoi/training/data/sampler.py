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

    def __init__(
        self,
        input_provider: "RecordProvider",
        target_provider: "RecordProvider",
    ) -> None:
        self.valid_time_indices = self.compute_valid_indices(input_provider, target_provider)

    @abstractmethod
    def compute_valid_indices(
        self,
        input_provider: "RecordProvider",
        target_provider: "RecordProvider",
    ) -> list[int]: ...


class AnemoiSampler(BaseAnemoiSampler):
    """Sampler"""

    def __init__(
        self,
        input_provider: "RecordProvider",
        target_provider: "RecordProvider",
        frequency: str,
        start: str | int,
        end: str | int,
    ):
        self.frequency = frequency
        self.start = parse_date(start)
        self.end = parse_date(end)

        super().__init__(input_provider, target_provider)

    @cached_property
    def time_values(self) -> np.array:
        """Time indices"""
        time_values = pd.date_range(start=self.start, end=self.end, freq=self.frequency)
        return np.array(time_values, dtype="datetime64[ns]")

    def compute_valid_indices(
        self,
        input_provider: "RecordProvider",
        target_provider: "RecordProvider",
    ) -> list[int]:
        """Set the valid indices.

        This method set the valid refernce indices for sampling.
        """
        # TODO: Handle missing data
        valid_time_indices = np.full(len(self.time_values), True)

        for sample_provider in [input_provider, target_provider]:
            coverages = sample_provider.get_sample_coverage()
            for dh_key, (start_date, end_date, freq_td) in coverages.items():
                steps = sample_provider._steps[dh_key]
                prev_steps, future_steps = min(steps), max(steps)

                freq = np.timedelta64(freq_td)
                start_date = np.datetime64(start_date)
                end_date = np.datetime64(end_date)
                min_valid_time = start_date + prev_steps * freq
                max_valid_time = end_date - future_steps * freq
                is_within_range = (self.time_values >= min_valid_time) & (self.time_values <= max_valid_time)
                valid_time_indices &= is_within_range

        return [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
        # return list(np.where(valid_time_indices)[0])
