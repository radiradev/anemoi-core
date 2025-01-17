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
from _pytest.fixtures import SubRequest
from hydra.utils import instantiate
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.train.forecaster import GraphForecaster
from anemoi.training.train.scaling import GeneralVariableLossScaler
from anemoi.training.train.scaling import NoTendencyScaler
from anemoi.training.train.scaling import StdevTendencyScaler
from anemoi.training.train.scaling import VarTendencyScaler


@pytest.fixture
def fake_data(request: SubRequest) -> tuple[DictConfig, IndexCollection]:
    config = DictConfig(
        {
            "data": {
                "forcing": ["x"],
                "diagnostic": ["z", "q"],
                "remapped": {
                    "d": ["cos_d", "sin_d"],
                },
            },
            "training": {
                "variable_loss_scaling": {
                    "variable_groups": {
                        "default": "sfc",
                        "pl": ["y"],
                    },
                    "default": 1,
                    "z": 0.1,
                    "other": 100,
                    "y": 0.5,
                },
                "metrics": ["other", "y_850"],
                "additional_scalars": request.param,
            },
        },
    )
    name_to_index = {"x": 0, "y_50": 1, "y_500": 2, "y_850": 3, "z": 5, "q": 4, "other": 6, "d": 7}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return config, data_indices


linear_scaler = [
    {
        "_target_": "anemoi.training.train.scaling.LinearVariableLevelScaler",
        "group": "pl",
        "y_intercept": 0.0,
        "slope": 0.001,
        "scale_dim": -1,
        "name": "variable_pressure_level",
    },
]
relu_scaler = [
    {
        "_target_": "anemoi.training.train.scaling.ReluVariableLevelScaler",
        "group": "pl",
        "y_intercept": 0.2,
        "slope": 0.001,
        "scale_dim": -1,
        "name": "variable_pressure_level",
    },
]
constant_scaler = [
    {
        "_target_": "anemoi.training.train.scaling.NoVariableLevelScaler",
        "group": "pl",
        "y_intercept": 1.0,
        "slope": 0.0,
        "scale_dim": -1,
        "name": "variable_pressure_level",
    },
]
polynomial_scaler = [
    {
        "_target_": "anemoi.training.train.scaling.PolynomialVariableLevelScaler",
        "group": "pl",
        "y_intercept": 0.2,
        "slope": 0.001,
        "scale_dim": -1,
        "name": "variable_pressure_level",
    },
]

std_dev_scaler = {
    "- _target_": "anemoi.training.train.scaling.StdevTendencyScaler",
    "scale_dim": -1,
    "name": "tendency",
}
var_scaler = {"- _target_": "anemoi.training.train.scaling.VarTendencyScaler", "scale_dim": -1, "name": "tendency"}

no_tend_scaler = {
    "- _target_": "anemoi.training.train.scaling.NoTendencyScaler",
    "scale_dim": -1,
    "name": "no_tendency",
}

expected_linear_scaling = torch.Tensor(
    [
        50 / 1000 * 0.5,  # y_50
        500 / 1000 * 0.5,  # y_500
        850 / 1000 * 0.5,  # y_850
        1,  # q
        0.1,  # z
        100,  # other
        1,  # cos_d
        1,  # sin_d
    ],
)
expected_relu_scaling = torch.Tensor(
    [
        0.2 * 0.5,  # y_50
        500 / 1000 * 0.5,  # y_500
        850 / 1000 * 0.5,  # y_850
        1,  # q
        0.1,  # z
        100,  # other
        1,  # cos_d
        1,  # sin_d
    ],
)
expected_constant_scaling = torch.Tensor(
    [
        1 * 0.5,  # y_50
        1 * 0.5,  # y_500
        1 * 0.5,  # y_850
        1,  # q
        0.1,  # z
        100,  # other
        1,  # cos_d
        1,  # sin_d
    ],
)
expected_polynomial_scaling = torch.Tensor(
    [
        ((50 / 1000) ** 2 + 0.2) * 0.5,  # y_50
        ((500 / 1000) ** 2 + 0.2) * 0.5,  # y_500
        ((850 / 1000) ** 2 + 0.2) * 0.5,  # y_850
        1,  # q
        0.1,  # z
        100,  # other
        1,  # cos_d
        1,  # sin_d
    ],
)


@pytest.mark.parametrize(
    ("fake_data", "expected_scaling"),
    [
        (linear_scaler, expected_linear_scaling),
        (relu_scaler, expected_relu_scaling),
        (constant_scaler, expected_constant_scaling),
        (polynomial_scaler, expected_polynomial_scaling),
    ],
    indirect=["fake_data"],
)
def test_variable_loss_scaling_vals(
    fake_data: tuple[DictConfig, IndexCollection],
    expected_scaling: torch.Tensor,
) -> None:
    config, data_indices = fake_data
    variable_scaling = GeneralVariableLossScaler(
        config.training.variable_loss_scaling,
        data_indices,
    ).get_variable_scaling()

    scalar = [
        (
            instantiate(
                scalar_config,
                scaling_config=config.training.variable_loss_scaling,
                data_indices=data_indices,
                statistics=None,
                statistics_tendencies=None,
            )
            if scalar_config["name"] == "tendency"
            else instantiate(
                scalar_config,
                scaling_config=config.training.variable_loss_scaling,
                data_indices=data_indices,
                metadata_variables=None,
            )
        )
        for scalar_config in config.training.additional_scalars
    ]

    scalars = {
        "variable": (-1, variable_scaling),
    }
    # add addtional user-defined scalars
    [scalars.update({scale.name: (scale.scale_dim, scale.get_variable_scaling())}) for scale in scalar]
    keys_list = list(scalars.keys())
    scalars[keys_list[0]][1] * scalars[keys_list[1]][1]

    assert torch.allclose(torch.tensor(scalars[keys_list[0]][1] * scalars[keys_list[1]][1]), expected_scaling)


@pytest.mark.parametrize("fake_data", [linear_scaler], indirect=["fake_data"])
def test_metric_range(fake_data: tuple[DictConfig, IndexCollection]) -> None:
    config, data_indices = fake_data

    metric_range, metric_ranges_validation = GraphForecaster.get_val_metric_ranges(config, data_indices)

    del metric_range["all"]
    del metric_ranges_validation["all"]

    expected_metric_range_validation = {
        "pl_y": [
            data_indices.model.output.name_to_index["y_50"],
            data_indices.model.output.name_to_index["y_500"],
            data_indices.model.output.name_to_index["y_850"],
        ],
        "sfc_other": [data_indices.model.output.name_to_index["other"]],
        "sfc_q": [data_indices.model.output.name_to_index["q"]],
        "sfc_z": [data_indices.model.output.name_to_index["z"]],
        "other": [data_indices.model.output.name_to_index["other"]],
        "sfc_d": [data_indices.model.output.name_to_index["d"]],
        "y_850": [data_indices.model.output.name_to_index["y_850"]],
    }

    expected_metric_range = expected_metric_range_validation.copy()
    del expected_metric_range["sfc_d"]
    expected_metric_range["sfc_cos_d"] = [data_indices.internal_model.output.name_to_index["cos_d"]]
    expected_metric_range["sfc_sin_d"] = [data_indices.internal_model.output.name_to_index["sin_d"]]

    assert metric_ranges_validation == expected_metric_range_validation
    assert metric_range == expected_metric_range


# TODO(Mariana): Add tests for the following classes
def test_no_tendency_scaling() -> None:
    scaler = NoTendencyScaler()
    result = scaler.get_level_scaling(10.0, 5.0)
    assert result == 1.0, "NoTendencyScaler should always return 1.0"


def test_stddev_tendency_scaling() -> None:
    scaler = StdevTendencyScaler()
    result = scaler.get_level_scaling(10.0, 5.0)
    expected = 10.0 / 5.0
    assert (
        pytest.approx(result, rel=1e-5) == expected
    ), "StdevTendencyScaler should return variable_stdev / variable_tendency_stdev"

    # Test with edge case
    result = scaler.get_level_scaling(0.0, 1.0)
    assert result == 0.0, "StdevTendencyScaler should return 0.0 when variable_stdev is 0"

    # Test division by a very small number
    result = scaler.get_level_scaling(1.0, 1e-6)
    expected = 1.0 / 1e-6
    assert pytest.approx(result, rel=1e-5) == expected, "StdevTendencyScaler should handle small divisor values"


def test_get_level_scaling() -> None:
    scaler = VarTendencyScaler()
    result = scaler.get_level_scaling(10.0, 5.0)
    expected = (10.0**2) / (5.0**2)
    assert (
        pytest.approx(result, rel=1e-5) == expected
    ), "VarTendencyScaler should return (variable_stdev^2) / (variable_tendency_stdev^2)"

    # Test with edge case
    result = scaler.get_level_scaling(0.0, 1.0)
    assert result == 0.0, "VarTendencyScaler should return 0.0 when variable_stdev is 0"

    # Test division by a very small number
    result = scaler.get_level_scaling(1.0, 1e-3)
    expected = (1.0**2) / (1e-3**2)
    assert pytest.approx(result, rel=1e-5) == expected, "VarTendencyScaler should handle small divisor values"
