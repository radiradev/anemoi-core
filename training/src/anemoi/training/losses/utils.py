# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from anemoi.training.utils.enums import TensorDim

if TYPE_CHECKING:
    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.losses.base import BaseLoss

LOGGER = logging.getLogger(__name__)


def print_variable_scaling(loss: BaseLoss, data_indices: IndexCollection) -> None:
    """Log the final variable scaling for each variable in the model.

    Parameters
    ----------
    loss : BaseLoss
        Loss function to get the variable scaling from.
    data_indices : IndexCollection
        Index collection to get the variable names from.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping variable names to their scaling values.
    """
    variable_scaling = loss.scaler.subset_by_dim(TensorDim.VARIABLE.value).get_scaler(len(TensorDim)).squeeze()
    log_text = "Final Variable Scaling: "
    scaling_values = {}

    for idx, name in enumerate(data_indices.internal_model.output.name_to_index.keys()):
        value = float(variable_scaling[idx])
        log_text += f"{name}: {value:.4g}, "
        scaling_values[name] = value

    LOGGER.debug(log_text)
    return scaling_values
