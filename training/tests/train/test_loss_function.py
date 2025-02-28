# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch
from omegaconf import DictConfig

from anemoi.training.losses.mse import WeightedMSELoss
from anemoi.training.losses.kcrps import KernelCRPS, AlmostFairKernelCRPS
from anemoi.training.losses.weightedloss import BaseWeightedLoss
from anemoi.training.train.forecaster import GraphForecaster
from anemoi.training.train.forecaster import GraphEnsForecaster


def test_manual_init() -> None:
    loss = WeightedMSELoss(torch.ones(1))
    assert loss.node_weights == torch.ones(1)


def test_dynamic_init_include() -> None:
    loss = GraphForecaster.get_loss_function(
        DictConfig({"_target_": "anemoi.training.losses.mse.WeightedMSELoss"}),
        node_weights=torch.ones(1),
    )
    assert isinstance(loss, BaseWeightedLoss)
    assert loss.node_weights == torch.ones(1)


def test_dynamic_init_scalar() -> None:
    loss = GraphForecaster.get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.mse.WeightedMSELoss",
                "scalars": ["test"],
            },
        ),
        node_weights=torch.ones(1),
        scalars={"test": ((0, 1), torch.ones((1, 2)))},
    )
    assert isinstance(loss, BaseWeightedLoss)

    torch.testing.assert_close(loss.node_weights, torch.ones(1))
    assert "test" in loss.scalar
    torch.testing.assert_close(loss.scalar.get_scalar(2), torch.ones((1, 2)))


def test_dynamic_init_scalar_not_add() -> None:
    loss = GraphForecaster.get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.mse.WeightedMSELoss",
                "scalars": [],
            },
        ),
        node_weights=torch.ones(1),
        scalars={"test": (-1, torch.ones(2))},
    )
    assert isinstance(loss, BaseWeightedLoss)
    torch.testing.assert_close(loss.node_weights, torch.ones(1))
    assert "test" not in loss.scalar

# KernelCRPS tests
def test_kcrps_manual_init() -> None:
    """Test manual initialization of KernelCRPS."""
    loss = KernelCRPS(torch.ones(1), fair=True)
    assert isinstance(loss, BaseWeightedLoss)
    assert loss.node_weights == torch.ones(1)
    assert loss.fair is True


def test_kcrps_dynamic_init() -> None:
    """Test dynamic initialization of KernelCRPS through config."""
    loss = GraphForecaster.get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.kcrps.KernelCRPS",
                "fair": True,
            }
        ),
        node_weights=torch.ones(1),
    )
    assert isinstance(loss, BaseWeightedLoss)
    assert loss.node_weights == torch.ones(1)
    assert loss.fair is True


def test_almost_fair_kcrps_manual_init() -> None:
    """Test manual initialization of AlmostFairKernelCRPS."""
    loss = AlmostFairKernelCRPS(torch.ones(1), alpha=0.95)
    assert isinstance(loss, BaseWeightedLoss)
    assert loss.node_weights == torch.ones(1)
    assert loss.alpha == 0.95


def test_almost_fair_kcrps_dynamic_init() -> None:
    """Test dynamic initialization of AlmostFairKernelCRPS through config."""
    loss = GraphForecaster.get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.kcrps.AlmostFairKernelCRPS",
                "alpha": 0.95,
            }
        ),
        node_weights=torch.ones(1),
    )
    assert isinstance(loss, BaseWeightedLoss)
    assert loss.node_weights == torch.ones(1)
    assert loss.alpha == 0.95


def test_kcrps_with_scalars() -> None:
    """Test KernelCRPS with scalar variables."""
    loss = GraphForecaster.get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.kcrps.KernelCRPS",
                "scalars": ["test"],
                "fair": True,
            }
        ),
        node_weights=torch.ones(1),
        scalars={"test": ((0, 1), torch.ones((1, 2)))},
    )
    assert isinstance(loss, BaseWeightedLoss)
    assert "test" in loss.scalar
    torch.testing.assert_close(loss.scalar.get_scalar(2), torch.ones((1, 2)))


def test_almost_fair_kcrps_with_scalars() -> None:
    """Test AlmostFairKernelCRPS with scalar variables."""
    loss = GraphForecaster.get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.kcrps.AlmostFairKernelCRPS",
                "scalars": ["test"],
                "alpha": 0.95,
            }
        ),
        node_weights=torch.ones(1),
        scalars={"test": ((0, 1), torch.ones((1, 2)))},
    )
    assert isinstance(loss, BaseWeightedLoss)
    assert "test" in loss.scalar
    torch.testing.assert_close(loss.scalar.get_scalar(2), torch.ones((1, 2)))