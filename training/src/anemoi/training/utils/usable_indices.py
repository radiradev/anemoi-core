# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations
from typing import Union

import numpy as np
import torch


def get_usable_indices(
    missing_indices: set[int] | None,
    series_length: int,
    rollout: int,
    multistep: int,
    timeincrement: int = 1,
) -> np.ndarray:
    """Get the usable indices of a series whit missing indices.

    Parameters
    ----------
    missing_indices : set[int]
        Dataset to be used.
    series_length : int
        Length of the series.
    rollout : int
        Number of steps to roll out.
    multistep : int
        Number of previous indices to include as predictors.
    timeincrement : int
        Time increment, by default 1.

    Returns
    -------
    usable_indices : np.array
        Array of usable indices.
    """
    prev_invalid_dates = (multistep - 1) * timeincrement
    next_invalid_dates = rollout * timeincrement

    usable_indices = np.arange(series_length)  # set of all indices

    if missing_indices is None:
        missing_indices = set()

    missing_indices |= {-1, series_length}  # to filter initial and final indices

    # Missing indices
    for i in missing_indices:
        usable_indices = usable_indices[
            (usable_indices < i - next_invalid_dates) + (usable_indices > i + prev_invalid_dates)
        ]

    return usable_indices

def try_get_slice(indices: torch.Tensor) -> Union[slice, torch.Tensor]:
    """Try to convert a tensor of indices into a slice.

    Parameters
    ----------
    indices: torch.Tensor
        Tensor of indices.
    
    Returns
    -------
    out: slice | torch.Tensor
        Slice if the indices are contiguous, otherwise the unmodified indices.
    """
    # could be done faster, assume performance is not an issue here
    min_index, max_index = indices.min().item(), indices.max().item()
    if indices.tolist() == list(range(min_index, max_index + 1)):
        return slice(min_index, max_index + 1)
    return indices