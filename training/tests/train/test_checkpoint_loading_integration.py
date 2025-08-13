# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import torch
import torch.nn as nn
from omegaconf import DictConfig

from anemoi.training.train.train import AnemoiTrainer


class SimpleMockModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class TestCheckpointLoadingIntegration:
    """Integration tests for checkpoint loading in the training pipeline."""

    def create_test_checkpoint(self, tmp_path: Path) -> Path:
        """Create a test checkpoint file."""
        model = SimpleMockModel()
        checkpoint_path = tmp_path / "test_checkpoint.ckpt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer": {},
                "lr_scheduler": {},
                "epoch": 1,
            },
            checkpoint_path,
        )
        return checkpoint_path

    @patch("anemoi.training.utils.model_loading.load_model_from_checkpoint")
    def test_load_checkpoint_if_configured_with_valid_config(self, mock_load_model, tmp_path):
        """Test _load_checkpoint_if_configured with valid configuration."""
        # Create test checkpoint
        checkpoint_path = self.create_test_checkpoint(tmp_path)

        # Mock trainer with minimal config
        config = DictConfig(
            {
                "training": {
                    "checkpoint_loading": {
                        "source": str(checkpoint_path),
                        "loader_type": "weights_only",
                        "strict": True,
                    },
                },
            },
        )

        trainer = AnemoiTrainer.__new__(AnemoiTrainer)
        trainer.config = config

        # Create mock model
        mock_model = SimpleMockModel()
        mock_load_model.return_value = mock_model

        # Test the method
        result = trainer._load_checkpoint_if_configured(mock_model)

        # Verify load_model_from_checkpoint was called
        mock_load_model.assert_called_once_with(
            model=mock_model,
            checkpoint_source=str(checkpoint_path),
            loader_type="weights_only",
            strict=True,
        )
        assert result == mock_model

    def test_load_checkpoint_if_configured_no_config(self):
        """Test _load_checkpoint_if_configured with no checkpoint_loading config."""
        config = DictConfig({"training": {}})

        trainer = AnemoiTrainer.__new__(AnemoiTrainer)
        trainer.config = config

        mock_model = SimpleMockModel()
        result = trainer._load_checkpoint_if_configured(mock_model)

        # Should return original model unchanged
        assert result is mock_model

    def test_load_checkpoint_if_configured_no_source(self):
        """Test _load_checkpoint_if_configured with config but no source."""
        config = DictConfig({"training": {"checkpoint_loading": {"loader_type": "weights_only"}}})

        trainer = AnemoiTrainer.__new__(AnemoiTrainer)
        trainer.config = config

        mock_model = SimpleMockModel()

        with patch("anemoi.training.train.train.LOGGER") as mock_logger:
            result = trainer._load_checkpoint_if_configured(mock_model)

            # Should log warning and return original model
            mock_logger.warning.assert_called_once()
            assert result is mock_model

    @patch("anemoi.training.utils.model_loading.load_model_from_checkpoint")
    def test_load_checkpoint_if_configured_with_optional_params(self, mock_load_model, tmp_path):
        """Test _load_checkpoint_if_configured with optional parameters."""
        checkpoint_path = self.create_test_checkpoint(tmp_path)

        config = DictConfig(
            {
                "training": {
                    "checkpoint_loading": {
                        "source": str(checkpoint_path),
                        "loader_type": "transfer_learning",
                        "strict": False,
                        "skip_mismatched": True,
                    },
                },
            },
        )

        trainer = AnemoiTrainer.__new__(AnemoiTrainer)
        trainer.config = config

        mock_model = SimpleMockModel()
        mock_load_model.return_value = mock_model

        trainer._load_checkpoint_if_configured(mock_model)

        # Verify all parameters were passed
        mock_load_model.assert_called_once_with(
            model=mock_model,
            checkpoint_source=str(checkpoint_path),
            loader_type="transfer_learning",
            strict=False,
            skip_mismatched=True,
        )

    def test_load_checkpoint_if_configured_missing_loader_type(self, tmp_path):
        """Test behavior with missing loader_type."""
        checkpoint_path = self.create_test_checkpoint(tmp_path)

        config = DictConfig(
            {
                "training": {
                    "checkpoint_loading": {
                        "source": str(checkpoint_path),
                        # missing loader_type
                    },
                },
            },
        )

        trainer = AnemoiTrainer.__new__(AnemoiTrainer)
        trainer.config = config

        mock_model = SimpleMockModel()

        # Should handle missing loader_type gracefully
        with patch("anemoi.training.utils.model_loading.load_model_from_checkpoint") as mock_load_model:
            trainer._load_checkpoint_if_configured(mock_model)

            # Should still be called, with None loader_type
            mock_load_model.assert_called_once()
            call_kwargs = mock_load_model.call_args.kwargs
            assert call_kwargs["checkpoint_source"] == str(checkpoint_path)

    @patch("anemoi.training.utils.model_loading.load_model_from_checkpoint")
    def test_load_checkpoint_integration_in_model_property(self, mock_load_model, tmp_path):
        """Test checkpoint loading integration in model property."""
        checkpoint_path = self.create_test_checkpoint(tmp_path)

        # Create minimal config with checkpoint loading
        config = DictConfig(
            {"training": {"checkpoint_loading": {"source": str(checkpoint_path), "loader_type": "weights_only"}}},
        )

        # Mock the trainer's dependencies
        with patch.multiple(
            AnemoiTrainer,
            config=config,
            data_indices=MagicMock(),
            graph_data=MagicMock(),
            metadata=MagicMock(),
            datamodule=MagicMock(),
            supporting_arrays=MagicMock(),
            load_weights_only=False,  # Ensure legacy loading doesn't interfere
        ):
            trainer = AnemoiTrainer.__new__(AnemoiTrainer)
            trainer.config = config
            trainer.data_indices = MagicMock()
            trainer.graph_data = MagicMock()
            trainer.metadata = MagicMock()
            trainer.datamodule = MagicMock()
            trainer.supporting_arrays = MagicMock()
            trainer.load_weights_only = False

            mock_model = SimpleMockModel()
            mock_load_model.return_value = mock_model

            # Mock GraphForecaster construction
            with patch("anemoi.training.train.train.GraphForecaster") as mock_forecaster:
                mock_forecaster.return_value = mock_model

                # Call model property
                result = trainer.model

                # Verify checkpoint loading was called
                mock_load_model.assert_called_once()
                assert result == mock_model

    def test_legacy_and_new_checkpoint_loading_priority(self, tmp_path):
        """Test that new checkpoint loading takes precedence over legacy."""
        checkpoint_path = self.create_test_checkpoint(tmp_path)

        config = DictConfig(
            {"training": {"checkpoint_loading": {"source": str(checkpoint_path), "loader_type": "weights_only"}}},
        )

        with patch.multiple(
            AnemoiTrainer,
            config=config,
            data_indices=MagicMock(),
            graph_data=MagicMock(),
            metadata=MagicMock(),
            datamodule=MagicMock(),
            supporting_arrays=MagicMock(),
            load_weights_only=True,  # Legacy loading would normally activate
            last_checkpoint=str(checkpoint_path),
        ):
            trainer = AnemoiTrainer.__new__(AnemoiTrainer)
            trainer.config = config
            trainer.load_weights_only = True
            trainer.last_checkpoint = str(checkpoint_path)

            with patch("anemoi.training.utils.model_loading.load_model_from_checkpoint") as mock_new_load:
                with patch("anemoi.training.train.train.GraphForecaster") as mock_forecaster:
                    mock_model = SimpleMockModel()
                    mock_forecaster.return_value = mock_model
                    mock_new_load.return_value = mock_model

                    trainer.model

                    # New checkpoint loading should be called
                    mock_new_load.assert_called_once()
                    # Legacy loading should not happen due to the condition
                    mock_forecaster.load_from_checkpoint.assert_not_called()
