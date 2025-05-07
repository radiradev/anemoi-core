# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.training.data.data_handlers import SampleProvider
from abc import ABC, abstractmethod
from anemoi.training.data.utils import parse_date, SamplerProviderName, DataHandlerName
import pandas as pd
from datetime import datetime
from functools import cached_property
import numpy as np
from datetime import timedelta


class BaseAnemoiSampler(ABC):
    """Base AnemoiSampler class"""
    def __init__(self) -> None:
        self.valid_time_indices = ValueError

    @abstractmethod
    def set_valid_indices(
        self, 
        sample_providers: dict[SamplerProviderName, SampleProvider]
    ) -> None:
        ...


class AnemoiSampler(BaseAnemoiSampler):
    """Sampler"""

    def __init__(self, frequency: str, start: str | int, end: str | int):
        super().__init__()
        self.frequency = frequency
        self.start = parse_date(start)
        self.end = parse_date(end)

    @cached_property
    def time_values(self) -> np.array:
        """Time indices"""
        time_values = pd.date_range(start=self.start, end=self.end, freq=self.frequency)
        return np.array(time_values, dtype='datetime64[ns]')

    def set_valid_indices(
        self, 
        sample_providers: dict[SamplerProviderName, SampleProvider]
    ) -> None:
        """Set the valid indices.

        This method set the valid refernce indices for sampling.

        Arguments
        ---------
        sample_providers : dict[SamplerProviderName, SampleProvider]
            Sample provider. For example, 
            ```
                {"input": SamplerProvider(...), "output": SamplerProvider(...)}
            ```
        """
        # TODO: Handle missing data
        valid_time_indices = np.full(len(self.time_values), True)

        for sp in sample_providers.values():
            coverages = sp.get_sample_coverage()
            for dh_key, (start_date, end_date, freq_td) in coverages.items():
                steps = sp._steps[dh_key]
                prev_steps, future_steps = min(steps), max(steps)

                freq = np.timedelta64(freq_td)
                min_valid_time = start_date + prev_steps * freq
                max_valid_time = end_date - future_steps * freq
                is_within_range = (self.time_values >= min_valid_time) & (self.time_values <= max_valid_time)
                valid_time_indices &= is_within_range

        self.valid_time_indices = list(np.where(valid_time_indices)[0])
