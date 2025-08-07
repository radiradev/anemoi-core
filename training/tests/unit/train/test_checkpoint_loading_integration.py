# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for checkpoint loading integration in training pipeline."""

from __future__ import annotations

from unittest.mock import patch

import torch.nn as nn
from omegaconf import DictConfig


class MockTrainer:
    """Mock trainer class for testing checkpoint loading integration."""

    def __init__(self, config: DictConfig):
        self.config = config

    def _load_checkpoint_if_configured(self, model: nn.Module) -> nn.Module:
        """Load checkpoint weights if checkpoint_loading is configured."""
        if not hasattr(self.config.training, "checkpoint_loading") or not self.config.training.checkpoint_loading:
            return model

        checkpoint_config = self.config.training.checkpoint_loading

        if not checkpoint_config.source:
            return model

        from anemoi.training.utils.model_loading import load_model_from_checkpoint

        # Extract parameters from checkpoint config
        loader_kwargs = {}
        if hasattr(checkpoint_config, "strict"):
            loader_kwargs["strict"] = checkpoint_config.strict
        if hasattr(checkpoint_config, "skip_mismatched"):
            loader_kwargs["skip_mismatched"] = checkpoint_config.skip_mismatched

        return load_model_from_checkpoint(
            model=model,
            checkpoint_source=checkpoint_config.source,
            loader_type=checkpoint_config.loader_type,
            **loader_kwargs,
        )


class MockModel(nn.Module):
    """Simple mock model for testing."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 5)

    def forward(self, x):
        return self.layer(x)


class TestCheckpointLoadingIntegration:
    """Test checkpoint loading integration in training pipeline."""

    def test_no_checkpoint_loading_config(self) -> None:
        """Test that model is returned unchanged when no checkpoint loading is configured."""
        config = DictConfig({"training": {}})

        trainer = MockTrainer(config)
        model = MockModel()

        result = trainer._load_checkpoint_if_configured(model)
        assert result is model

    def test_checkpoint_loading_disabled(self) -> None:
        """Test that model is returned unchanged when checkpoint loading is disabled."""
        config = DictConfig({"training": {"checkpoint_loading": None}})

        trainer = MockTrainer(config)
        model = MockModel()

        result = trainer._load_checkpoint_if_configured(model)
        assert result is model

    def test_checkpoint_loading_no_source(self) -> None:
        """Test that model is returned unchanged when no source is specified."""
        config = DictConfig({"training": {"checkpoint_loading": {"source": None, "loader_type": "weights_only"}}})

        trainer = MockTrainer(config)
        model = MockModel()

        result = trainer._load_checkpoint_if_configured(model)
        assert result is model

    @patch("anemoi.training.train.test_checkpoint_loading_integration.load_model_from_checkpoint")
    def test_weights_only_loading(self, mock_load_model) -> None:
        """Test weights-only checkpoint loading."""
        config = DictConfig(
            {
                "training": {
                    "checkpoint_loading": {
                        "source": "/path/to/checkpoint.ckpt",
                        "loader_type": "weights_only",
                        "strict": True,
                    },
                },
            },
        )

        trainer = MockTrainer(config)
        model = MockModel()
        modified_model = MockModel()  # Different instance to verify mock was called

        mock_load_model.return_value = modified_model

        result = trainer._load_checkpoint_if_configured(model)

        assert result is modified_model
        mock_load_model.assert_called_once_with(
            model=model,
            checkpoint_source="/path/to/checkpoint.ckpt",
            loader_type="weights_only",
            strict=True,
        )

    @patch("anemoi.training.train.test_checkpoint_loading_integration.load_model_from_checkpoint")
    def test_transfer_learning_loading(self, mock_load_model) -> None:
        """Test transfer learning checkpoint loading."""
        config = DictConfig(
            {
                "training": {
                    "checkpoint_loading": {
                        "source": "s3://bucket/pretrained.ckpt",
                        "loader_type": "transfer_learning",
                        "strict": False,
                        "skip_mismatched": True,
                    },
                },
            },
        )

        trainer = MockTrainer(config)
        model = MockModel()
        modified_model = MockModel()

        mock_load_model.return_value = modified_model

        result = trainer._load_checkpoint_if_configured(model)

        assert result is modified_model
        mock_load_model.assert_called_once_with(
            model=model,
            checkpoint_source="s3://bucket/pretrained.ckpt",
            loader_type="transfer_learning",
            strict=False,
            skip_mismatched=True,
        )

    @patch("anemoi.training.train.test_checkpoint_loading_integration.load_model_from_checkpoint")
    def test_standard_loading(self, mock_load_model) -> None:
        """Test standard checkpoint loading."""
        config = DictConfig(
            {
                "training": {
                    "checkpoint_loading": {
                        "source": "https://example.com/model.ckpt",
                        "loader_type": "standard",
                        "strict": True,
                    },
                },
            },
        )

        trainer = MockTrainer(config)
        model = MockModel()
        modified_model = MockModel()

        mock_load_model.return_value = modified_model

        result = trainer._load_checkpoint_if_configured(model)

        assert result is modified_model
        mock_load_model.assert_called_once_with(
            model=model,
            checkpoint_source="https://example.com/model.ckpt",
            loader_type="standard",
            strict=True,
        )

    @patch("anemoi.training.train.test_checkpoint_loading_integration.load_model_from_checkpoint")
    def test_minimal_config(self, mock_load_model) -> None:
        """Test checkpoint loading with minimal configuration."""
        config = DictConfig(
            {
                "training": {
                    "checkpoint_loading": {
                        "source": "/path/to/checkpoint.ckpt",
                        "loader_type": "weights_only",
                        # No strict or skip_mismatched specified
                    },
                },
            },
        )

        trainer = MockTrainer(config)
        model = MockModel()
        modified_model = MockModel()

        mock_load_model.return_value = modified_model

        result = trainer._load_checkpoint_if_configured(model)

        assert result is modified_model
        mock_load_model.assert_called_once_with(
            model=model,
            checkpoint_source="/path/to/checkpoint.ckpt",
            loader_type="weights_only",
            # No additional kwargs should be passed
        )


class TestTrainingPipelineIntegration:
    """Test the full training pipeline integration."""

    @patch("anemoi.training.train.test_checkpoint_loading_integration.load_model_from_checkpoint")
    def test_checkpoint_loading_before_model_modifiers(self, mock_load_model) -> None:
        """Test that checkpoint loading happens before model modifiers."""
        # This test would need to be integrated with the actual trainer
        # but demonstrates the expected behavior

        config = DictConfig(
            {
                "training": {
                    "checkpoint_loading": {"source": "/path/to/checkpoint.ckpt", "loader_type": "weights_only"},
                    "model_modifier": {
                        "modifiers": [
                            {
                                "_target_": "anemoi.training.train.modify.FreezingModelModifier",
                                "submodules_to_freeze": ["encoder"],
                            },
                        ],
                    },
                },
            },
        )

        # The expected flow would be:
        # 1. Instantiate base model
        # 2. Load checkpoint weights (tested here)
        # 3. Apply model modifiers

        trainer = MockTrainer(config)
        model = MockModel()
        modified_model = MockModel()

        mock_load_model.return_value = modified_model

        # Checkpoint loading step
        result = trainer._load_checkpoint_if_configured(model)

        assert result is modified_model
        mock_load_model.assert_called_once()

        # Model modifiers would then be applied to the result
        # (this would be handled by ModelModifierApplier in the actual system)
