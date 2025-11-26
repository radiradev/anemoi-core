import unittest
from unittest.mock import MagicMock, patch
import torch
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.interface import AnemoiModelInterface
from anemoi.training.train.tasks.forecaster import GraphForecaster
from anemoi.training.losses.mae import MAELoss
from anemoi.utils.config import DotDict
from anemoi.models.preprocessing import Processors
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# Mock torch.nn.Module components
class MockModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Add some parameters to avoid errors with optimizers
        self.layer = torch.nn.Linear(1, 1)

    def forward(self, *args, **kwargs):
        # A minimal forward pass
        if isinstance(args[0], tuple):
            return args[0][0], args[0][1]
        return args[0]

class TestRefactoredConfig(unittest.TestCase):
    @patch('anemoi.models.models.base.NamedNodesAttributes')
    def test_instantiation(self, mock_named_nodes_attributes):
        # This test will demonstrate direct instantiation of components.

        # 1. Create dependencies
        config = DotDict({
            "graph": {"data": "data", "hidden": "hidden"},
            "training": {"multistep_input": 1},
            "model": {"num_channels": 128, "trainable_parameters": {"hidden": 8}}
        })

        graph_data = MagicMock()
        graph_data.__getitem__.return_value = {'x': torch.randn(10, 2)} # Mock node features

        statistics = {'mean': torch.randn(1), 'stdev': torch.randn(1)}

        data_indices = MagicMock()
        data_indices.model.input.prognostic = [0]
        data_indices.model.output.prognostic = [0]
        data_indices.model.input.full = [0]
        data_indices.model.output.full = [0]
        data_indices.model.output.diagnostic = []

        mock_node_attributes = MagicMock()
        mock_node_attributes.attr_ndims = {"data": 0, "hidden": 0}
        mock_named_nodes_attributes.return_value = mock_node_attributes

        # 2. Instantiate components
        model = AnemoiModelEncProcDec(
            encoder=MockModule,
            processor=MockModule,
            decoder=MockModule,
            residual=MockModule,
            boundings=lambda **kwargs: [],
            model_config=config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
        )

        model_interface = AnemoiModelInterface(
            model=model,
            pre_processors=Processors([]),
            post_processors=Processors([]),
            multi_step=1,
        )

        graph_forecaster = GraphForecaster(
            model=model_interface.model,
            loss=MAELoss(),
            metrics={"mae": MAELoss()},
            optimizer_callable=lambda params: Adam(params, lr=0.001),
            lr_scheduler_callable=lambda optimizer: StepLR(optimizer, step_size=1),
            pre_processors=Processors([]),
            post_processors=Processors([]),
            multi_step=1,
        )

        # 3. Assert that objects were created successfully
        self.assertIsInstance(graph_forecaster, GraphForecaster)
        print("Successfully instantiated GraphForecaster.")

if __name__ == '__main__':
    unittest.main()
