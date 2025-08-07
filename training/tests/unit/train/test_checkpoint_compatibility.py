# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig

from anemoi.training.train.modify import TransferLearningModelModifier
from anemoi.training.train.modify import WeightsInitModelModifier


class TestModelArchitecture(nn.Module):
    """Test model with known architecture for checkpoint compatibility testing."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.processor = nn.ModuleList(
            [
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            ],
        )
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.weights_initialized = False
        self.skip_checkpoint_loading = False

    def forward(self, x):
        x = self.encoder(x)
        for layer in self.processor:
            x = layer(x)
        return self.decoder(x)


class TestCheckpointCompatibility:
    """Test checkpoint loading compatibility across different scenarios."""

    def create_checkpoint(
        self,
        temp_dir: Path,
        model: nn.Module,
        filename: str = "test_checkpoint.ckpt",
        include_data_indices: bool = True,
        include_hyper_parameters: bool = True,
    ) -> Path:
        """Create a checkpoint file from a model."""
        checkpoint_path = temp_dir / filename

        checkpoint_data = {"state_dict": model.state_dict()}

        if include_hyper_parameters:
            checkpoint_data["hyper_parameters"] = {}

            if include_data_indices:
                mock_data_indices = MagicMock()
                mock_data_indices.name_to_index = {"temperature": 0, "pressure": 1, "humidity": 2}
                checkpoint_data["hyper_parameters"]["data_indices"] = mock_data_indices

        torch.save(checkpoint_data, checkpoint_path)
        return checkpoint_path

    def test_exact_architecture_match(self) -> None:
        """Test loading checkpoint with exactly matching architecture."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create original model and checkpoint
            original_model = TestModelArchitecture(10, 20, 5)
            checkpoint_path = self.create_checkpoint(temp_path, original_model)

            # Create identical model for loading
            target_model = TestModelArchitecture(10, 20, 5)

            # Store original weights to verify loading
            original_encoder_weight = original_model.encoder.weight.clone()

            modifier = WeightsInitModelModifier(str(checkpoint_path))
            config = DictConfig({})

            result = modifier.apply(target_model, config)

            assert result is target_model
            assert target_model.weights_initialized is True
            assert torch.equal(target_model.encoder.weight, original_encoder_weight)

    def test_partial_architecture_mismatch_transfer_learning(self) -> None:
        """Test transfer learning with partially mismatched architecture."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create source model with different output dimension
            source_model = TestModelArchitecture(10, 20, 8)  # Different output dimension
            checkpoint_path = self.create_checkpoint(temp_path, source_model)

            # Create target model with different output
            target_model = TestModelArchitecture(10, 20, 5)  # Different output dimension

            # Store original encoder weight to verify partial loading
            original_encoder_weight = source_model.encoder.weight.clone()

            modifier = TransferLearningModelModifier(str(checkpoint_path))
            config = DictConfig({})

            # Should not raise exception
            result = modifier.apply(target_model, config)

            assert result is target_model
            assert target_model.skip_checkpoint_loading is True
            # Encoder should be loaded (compatible), decoder should be skipped (incompatible)
            assert torch.equal(target_model.encoder.weight, original_encoder_weight)

    def test_major_architecture_mismatch(self) -> None:
        """Test behavior with major architecture differences."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create source model
            source_model = TestModelArchitecture(10, 20, 5)
            checkpoint_path = self.create_checkpoint(temp_path, source_model)

            # Create completely different target model
            class DifferentModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.different_layer = nn.Linear(15, 25)
                    self.another_layer = nn.Conv1d(1, 1, 3)
                    self.skip_checkpoint_loading = False

                def forward(self, x):
                    return self.different_layer(x)

            target_model = DifferentModel()

            modifier = TransferLearningModelModifier(str(checkpoint_path))
            config = DictConfig({})

            # Should handle gracefully - no matching layers
            result = modifier.apply(target_model, config)

            assert result is target_model
            assert target_model.skip_checkpoint_loading is True

    def test_missing_checkpoint_file(self) -> None:
        """Test handling of missing checkpoint files."""
        modifier = WeightsInitModelModifier("/nonexistent/path/checkpoint.ckpt")
        model = TestModelArchitecture()
        config = DictConfig({})

        with pytest.raises(FileNotFoundError):
            modifier.apply(model, config)

    def test_corrupted_checkpoint_file(self) -> None:
        """Test handling of corrupted checkpoint files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "corrupted_checkpoint.ckpt"

            # Create corrupted checkpoint (not a valid PyTorch checkpoint)
            with open(checkpoint_path, "w") as f:
                f.write("This is not a valid checkpoint file")

            modifier = WeightsInitModelModifier(str(checkpoint_path))
            model = TestModelArchitecture()
            config = DictConfig({})

            with pytest.raises(Exception):  # Could be various exceptions
                modifier.apply(model, config)

    def test_checkpoint_missing_state_dict(self) -> None:
        """Test handling of checkpoint missing state_dict."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "invalid_checkpoint.ckpt"

            # Create checkpoint without state_dict
            invalid_checkpoint = {
                "epoch": 10,
                "optimizer_state": {},
                "lr_scheduler_state": {},
                # Missing state_dict
            }

            torch.save(invalid_checkpoint, checkpoint_path)

            modifier = WeightsInitModelModifier(str(checkpoint_path))
            model = TestModelArchitecture()
            config = DictConfig({})

            with pytest.raises(ValueError, match="No 'state_dict' found in checkpoint"):
                modifier.apply(model, config)

    def test_checkpoint_with_extra_layers(self) -> None:
        """Test loading checkpoint that has extra layers not in target model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create source model with extra layers
            class ExtendedModel(TestModelArchitecture):
                def __init__(self):
                    super().__init__()
                    self.extra_layer = nn.Linear(5, 3)
                    self.another_extra = nn.BatchNorm1d(20)

            source_model = ExtendedModel()
            checkpoint_path = self.create_checkpoint(temp_path, source_model)

            # Create simpler target model
            target_model = TestModelArchitecture()

            modifier = TransferLearningModelModifier(str(checkpoint_path))
            config = DictConfig({})

            # Should load successfully, ignoring extra layers
            result = modifier.apply(target_model, config)

            assert result is target_model
            assert target_model.skip_checkpoint_loading is True

    def test_checkpoint_with_missing_layers(self) -> None:
        """Test loading checkpoint that's missing layers present in target model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create simple source model
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.simple_layer = nn.Linear(10, 5)

                def forward(self, x):
                    return self.simple_layer(x)

            source_model = SimpleModel()
            checkpoint_path = self.create_checkpoint(temp_path, source_model)

            # Create complex target model
            target_model = TestModelArchitecture()

            modifier = TransferLearningModelModifier(str(checkpoint_path))
            config = DictConfig({})

            # Should load successfully, leaving unmatched layers with original weights
            result = modifier.apply(target_model, config)

            assert result is target_model
            assert target_model.skip_checkpoint_loading is True

    def test_data_indices_compatibility(self) -> None:
        """Test data indices handling and compatibility checking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create checkpoint with data indices
            source_model = TestModelArchitecture()
            checkpoint_path = self.create_checkpoint(temp_path, source_model)

            # Create target model with data indices
            target_model = TestModelArchitecture()
            target_model.data_indices = MagicMock()
            target_model.data_indices.compare_variables = MagicMock()
            target_model.data_indices.name_to_index = {"temperature": 0, "pressure": 1}

            modifier = WeightsInitModelModifier(str(checkpoint_path))
            config = DictConfig({})

            result = modifier.apply(target_model, config)

            assert result is target_model
            assert hasattr(target_model, "_ckpt_model_name_to_index")
            # Should call compare_variables for compatibility checking
            target_model.data_indices.compare_variables.assert_called_once()

    def test_checkpoint_without_data_indices(self) -> None:
        """Test handling checkpoint without data indices."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            source_model = TestModelArchitecture()
            checkpoint_path = self.create_checkpoint(temp_path, source_model, include_data_indices=False)

            target_model = TestModelArchitecture()

            modifier = WeightsInitModelModifier(str(checkpoint_path))
            config = DictConfig({})

            # Should work without data indices
            result = modifier.apply(target_model, config)

            assert result is target_model
            assert target_model.weights_initialized is True

    def test_checkpoint_without_hyper_parameters(self) -> None:
        """Test handling checkpoint without hyper_parameters section."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            source_model = TestModelArchitecture()
            checkpoint_path = self.create_checkpoint(temp_path, source_model, include_hyper_parameters=False)

            target_model = TestModelArchitecture()

            modifier = WeightsInitModelModifier(str(checkpoint_path))
            config = DictConfig({})

            # Should work without hyper_parameters
            result = modifier.apply(target_model, config)

            assert result is target_model
            assert target_model.weights_initialized is True


class TestTransferLearningEdgeCases:
    """Test edge cases specific to transfer learning."""

    def test_device_mismatch_handling(self) -> None:
        """Test transfer learning with device mismatches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create checkpoint
            source_model = TestModelArchitecture()
            checkpoint_path = temp_path / "checkpoint.ckpt"

            checkpoint_data = {
                "state_dict": source_model.state_dict(),
                "hyper_parameters": {"data_indices": MagicMock()},
            }
            checkpoint_data["hyper_parameters"]["data_indices"].name_to_index = {"test": 0}
            torch.save(checkpoint_data, checkpoint_path)

            # Create target model potentially on different device
            target_model = TestModelArchitecture()

            modifier = TransferLearningModelModifier(str(checkpoint_path))
            config = DictConfig({})

            # Should handle device mapping automatically
            result = modifier.apply(target_model, config)

            assert result is target_model
            assert target_model.skip_checkpoint_loading is True

    def test_dtype_mismatch_handling(self) -> None:
        """Test transfer learning with dtype mismatches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create source model with specific dtype
            source_model = TestModelArchitecture()
            source_model.encoder.weight.data = source_model.encoder.weight.data.to(torch.float64)

            checkpoint_path = temp_path / "checkpoint.ckpt"
            checkpoint_data = {
                "state_dict": source_model.state_dict(),
                "hyper_parameters": {"data_indices": MagicMock()},
            }
            checkpoint_data["hyper_parameters"]["data_indices"].name_to_index = {"test": 0}
            torch.save(checkpoint_data, checkpoint_path)

            # Create target model with different dtype
            target_model = TestModelArchitecture()
            target_model = target_model.to(torch.float32)

            modifier = TransferLearningModelModifier(str(checkpoint_path))
            config = DictConfig({})

            # Should handle dtype conversion
            result = modifier.apply(target_model, config)

            assert result is target_model
            assert target_model.skip_checkpoint_loading is True

    def test_parameter_name_variations(self) -> None:
        """Test handling of parameter names with variations (e.g., prefixes)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create source model
            source_model = TestModelArchitecture()

            # Manually create state dict with prefixed names (common in distributed training)
            state_dict = source_model.state_dict()
            prefixed_state_dict = {f"module.{k}": v for k, v in state_dict.items()}

            checkpoint_data = {"state_dict": prefixed_state_dict, "hyper_parameters": {"data_indices": MagicMock()}}
            checkpoint_data["hyper_parameters"]["data_indices"].name_to_index = {"test": 0}

            checkpoint_path = temp_path / "prefixed_checkpoint.ckpt"
            torch.save(checkpoint_data, checkpoint_path)

            # Create target model without prefixes
            target_model = TestModelArchitecture()

            modifier = TransferLearningModelModifier(str(checkpoint_path))
            config = DictConfig({})

            # Should handle name mismatches gracefully (no matching keys)
            result = modifier.apply(target_model, config)

            assert result is target_model
            assert target_model.skip_checkpoint_loading is True

    @patch("anemoi.training.utils.checkpoint.transfer_learning_loading")
    def test_transfer_learning_function_error_handling(self, mock_transfer_loading) -> None:
        """Test error handling in the underlying transfer learning function."""
        # Mock the transfer learning function to raise an exception
        mock_transfer_loading.side_effect = RuntimeError("Transfer learning failed")

        modifier = TransferLearningModelModifier("/path/to/checkpoint.ckpt")
        model = TestModelArchitecture()
        config = DictConfig({})

        with pytest.raises(RuntimeError, match="Transfer learning failed"):
            modifier.apply(model, config)


class TestWeightsInitEdgeCases:
    """Test edge cases specific to weights initialization."""

    def test_strict_loading_behavior(self) -> None:
        """Test that non-strict loading is used and handles mismatches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create checkpoint with extra parameters
            source_model = TestModelArchitecture()
            state_dict = source_model.state_dict()
            state_dict["extra_param"] = torch.randn(10, 10)  # Extra parameter

            checkpoint_data = {"state_dict": state_dict, "hyper_parameters": {"data_indices": MagicMock()}}
            checkpoint_data["hyper_parameters"]["data_indices"].name_to_index = {"test": 0}

            checkpoint_path = temp_path / "checkpoint_with_extra.ckpt"
            torch.save(checkpoint_data, checkpoint_path)

            target_model = TestModelArchitecture()

            modifier = WeightsInitModelModifier(str(checkpoint_path))
            config = DictConfig({})

            # Should load successfully with strict=False (ignoring extra parameters)
            result = modifier.apply(target_model, config)

            assert result is target_model
            assert target_model.weights_initialized is True

    def test_checkpoint_path_types(self) -> None:
        """Test that different path types (str, Path) work correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            source_model = TestModelArchitecture()
            checkpoint_path = temp_path / "checkpoint.ckpt"

            checkpoint_data = {
                "state_dict": source_model.state_dict(),
                "hyper_parameters": {"data_indices": MagicMock()},
            }
            torch.save(checkpoint_data, checkpoint_path)

            target_model = TestModelArchitecture()

            # Test with Path object
            modifier1 = WeightsInitModelModifier(checkpoint_path)
            result1 = modifier1.apply(target_model, DictConfig({}))
            assert result1.weights_initialized is True

            # Reset model
            target_model.weights_initialized = False

            # Test with string path
            modifier2 = WeightsInitModelModifier(str(checkpoint_path))
            result2 = modifier2.apply(target_model, DictConfig({}))
            assert result2.weights_initialized is True

    def test_checkpoint_loading_flags(self) -> None:
        """Test that proper flags are set for PyTorch Lightning integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            source_model = TestModelArchitecture()
            checkpoint_path = temp_path / "checkpoint.ckpt"

            checkpoint_data = {
                "state_dict": source_model.state_dict(),
                "hyper_parameters": {"data_indices": MagicMock()},
            }
            torch.save(checkpoint_data, checkpoint_path)

            target_model = TestModelArchitecture()

            # Verify initial state
            assert target_model.weights_initialized is False
            assert target_model.skip_checkpoint_loading is False

            modifier = WeightsInitModelModifier(str(checkpoint_path))
            config = DictConfig({})

            result = modifier.apply(target_model, config)

            # Verify flags are set correctly
            assert result.weights_initialized is True
            assert result.skip_checkpoint_loading is True
