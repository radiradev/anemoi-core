# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch.nn as nn
from torch import Tensor

class DictLoss(nn.Module):
    """Wrapper for a dictionary of loss-fuctions that operate on different outputs."""

    def __init__(
        self, 
        loss_dict: nn.ModuleDict,
    ) -> None:
        super().__init__()
        self.loss_dict = loss_dict
        self.outputs = list(loss_dict.keys())

    def forward(
        self,
        pred: dict[str, Tensor],
        target: dict[str, Tensor],
        squash: bool = True, # TODO Generalise this per output?
        ) -> dict[str, Tensor]:
        out = {}
        for output, loss in self.loss_dict.items():
            out[output] = loss(pred[output], target[output], squash)

        return out