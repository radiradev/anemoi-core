import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.models.layers.residual import TruncationMapper


@pytest.fixture
def graph_data():
    g = HeteroData()
    g["hidden", "to", "data"].edge_index = torch.tensor([[0, 0], [0, 1]])
    g["hidden", "to", "data"].edge_length = torch.tensor([1.0, 2.0])
    g["data", "to", "hidden"].edge_index = torch.tensor([[0, 1], [0, 0]])
    g["data", "to", "hidden"].edge_length = torch.tensor([1.0, 2.0])
    return {
        "sub_graph_down": g["data", "to", "hidden"],
        "sub_graph_up": g["hidden", "to", "data"],
        "num_data_nodes": 2,
        "num_truncation_nodes": 1,
    }


@pytest.fixture
def flat_data():
    x = torch.randn(11, 7, 5, 2, 3)  # batch, dates, ensemble, grid, features
    return x


@pytest.fixture
def edge_index():
    return torch.tensor([[0, 1, 1], [1, 0, 2]])


def test_truncation_mapper_init(graph_data):
    _ = TruncationMapper(**graph_data)


def test_forward(graph_data):
    mapper = TruncationMapper(**graph_data)
    x = torch.randn(5, 2, 2, 2, 3)  # (batch, dates, ensemble, grid, features)
    x_truncated = mapper.forward(x)
    assert x_truncated.shape == (5, 2, 2, 3)  # (batch, ensemble, coarse_grid, features)
