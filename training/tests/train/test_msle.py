# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from torch.testing import assert_close

from anemoi.training.losses.msle import WeightedMSLELoss


@pytest.fixture
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def basic_inputs(device: str) -> tuple:
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True, device=device)
    target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], device=device)
    node_weights = torch.ones([2], device=device)
    return pred, target, node_weights


def test_forward_pass(basic_inputs: torch.Tensor) -> None:
    pred, target, node_weights = basic_inputs
    loss_function = WeightedMSLELoss(node_weights)
    loss = loss_function(pred, target)

    assert isinstance(loss, torch.Tensor)
    assert_close(loss, torch.tensor(0.0, device=loss.device))


def test_weighted_forward_pass(device: str) -> None:
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], device=device, requires_grad=True)
    target = torch.tensor([[[[2.0, 2.0], [3.0, 4.0]]]], device=device)
    node_weights = torch.tensor([1.0, 2.0], device=device)
    loss_function = WeightedMSLELoss(node_weights)
    loss = loss_function(pred, target)

    assert loss.item() > 0
    loss.backward()
    assert pred.grad is not None
