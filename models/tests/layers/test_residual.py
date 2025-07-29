import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.models.layers.residual import TruncationMapper

#  from anemoi.models.layers.residual import hans


@pytest.fixture
def up_down_graph():
    graph = HeteroData()
    x_fine = torch.tensor([[0, 0], [0, 1]])
    x_coarse = torch.tensor([[0, 0.3]])

    edge_index_down = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)  # fine → coarse
    edge_index_up = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)  # coarse → fine

    graph["fine"].x = x_fine
    graph["coarse"].x = x_coarse
    graph["coarse", "to", "fine"].edge_index = edge_index_up
    graph["fine", "to", "coarse"].edge_index = edge_index_down

    return graph


@pytest.fixture
def flat_data():
    x = torch.randn(3, 2, 5)  # batch, node, features
    return x


@pytest.fixture
def edge_index():
    return torch.tensor([[0, 1, 1], [1, 0, 2]])


#  def test_truncation_matrix_from_edge_index(edge_index):
#      edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]])
#      edge_attribute = torch.tensor([1.0, 2.0, 3.0])
#      num_source_nodes = 2
#      num_target_nodes = 3
#
#      A = TruncationMapper._create_truncation_matrix(edge_index, edge_attribute, num_source_nodes, num_target_nodes)
#
#      assert A.is_sparse
#      assert A.shape == (2, 3)
#      indices = A.indices()
#      values = A.values()
#
#      expected_indices = torch.tensor([[0, 1, 1], [1, 0, 2]])
#      expected_values = torch.tensor([1.0, 2.0, 3.0])
#
#      assert torch.equal(indices, expected_indices)
#      assert torch.equal(values, expected_values)


def test_truncation_mapper_init(up_down_graph):
    mapper = TruncationMapper(up_down_graph)
    assert hasattr(mapper, "A_up")
    assert hasattr(mapper, "A_down")


def test_truncate_fields(up_down_graph, flat_data):
    mapper = TruncationMapper(up_down_graph)
    mapper._truncate_fields(flat_data, mapper.A_down)
    #  assert hasattr(mapper, "A_up")


#  def test_truncation_mapper_forward(up_down_graph, flat_data):
#      mapper = TruncationMapper(up_down_graph)
#      mapper(flat_data)
#      #  up_down_graph["fine", "to", "coarse"]
