# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing.postprocessor import Postprocessor


@pytest.fixture()
def postprocessor():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "postprocessor": {"default": "none", "relu": ["q"], "hardtanh": ["x"]},
                "forcing": ["z", "q"],
                "diagnostic": ["other"],
                "remapped": {},
            },
        },
    )
    statistics = {
        "mean": np.array([1.0, 2.0, 3.0, 4.5, 3.0]),
        "stdev": np.array([0.5, 0.5, 0.5, 1, 14]),
        "minimum": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "maximum": np.array([11.0, 10.0, 10.0, 10.0, 10.0]),
    }
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return Postprocessor(config=config.data.postprocessor, data_indices=data_indices, statistics=statistics)


@pytest.fixture()
def output_data():
    base = torch.Tensor([[1.0, 2.0, 3.0, -1, 5.0], [1.0, -2, 8.0, 9.0, 10.0]])
    expected = torch.Tensor([[1.0, 2.0, 3.0, 0.0, 5.0], [1.0, -1, 8.0, 9.0, 10.0]])
    return base, expected


fixture_combinations = (("postprocessor", "output_data"),)


@pytest.mark.parametrize(
    ("postprocessor_fixture", "data_fixture"),
    fixture_combinations,
)
def test_postprocessor_not_inplace(postprocessor_fixture, data_fixture, request) -> None:
    """Check that the imputer does not modify the input tensor when in_place=False."""
    x, _ = request.getfixturevalue(data_fixture)
    postprocessor = request.getfixturevalue(postprocessor_fixture)
    x_old = x.clone()
    postprocessor.inverse_transform(x, in_place=False)
    assert torch.allclose(x, x_old, equal_nan=True), "Postprocessor does not handle in_place=False correctly."


@pytest.mark.parametrize(
    ("postprocessor_fixture", "data_fixture"),
    fixture_combinations,
)
def test_postprocessor_inplace(postprocessor_fixture, data_fixture, request) -> None:
    """Check that the imputer does not modify the input tensor when in_place=False."""
    x, _ = request.getfixturevalue(data_fixture)
    postprocessor = request.getfixturevalue(postprocessor_fixture)
    x_old = x.clone()
    out = postprocessor.inverse_transform(x, in_place=True)
    assert not torch.allclose(x, x_old, equal_nan=True)
    assert torch.allclose(x, out, equal_nan=True)
