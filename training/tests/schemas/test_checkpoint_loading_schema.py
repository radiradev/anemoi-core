# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from omegaconf import DictConfig
from pydantic import ValidationError

from anemoi.training.schemas.base_schema import BaseSchema


class TestCheckpointLoadingSchema:
    """Test checkpoint_loading field validation in training schema."""

    def test_checkpoint_loading_field_exists(self):
        """Test that checkpoint_loading field exists in schema."""
        config_dict = {
            "training": {"checkpoint_loading": {"source": "/path/to/checkpoint.ckpt", "loader_type": "weights_only"}},
        }

        # Should not raise validation error
        config = BaseSchema(**DictConfig(config_dict))
        assert hasattr(config.training, "checkpoint_loading")
        assert config.training.checkpoint_loading is not None

    def test_checkpoint_loading_none_by_default(self):
        """Test that checkpoint_loading defaults to None."""
        config_dict = {"training": {}}

        config = BaseSchema(**DictConfig(config_dict))
        assert config.training.checkpoint_loading is None

    def test_checkpoint_loading_valid_config(self):
        """Test valid checkpoint_loading configurations."""
        valid_configs = [
            # Basic weights_only
            {"source": "/local/path/checkpoint.ckpt", "loader_type": "weights_only", "strict": True},
            # Transfer learning with options
            {
                "source": "s3://bucket/checkpoint.ckpt",
                "loader_type": "transfer_learning",
                "strict": False,
                "skip_mismatched": True,
            },
            # Standard loading
            {"source": "https://example.com/checkpoint.ckpt", "loader_type": "standard"},
            # Cloud sources
            {"source": "gs://bucket/checkpoint.ckpt", "loader_type": "weights_only"},
            {
                "source": "azure://account.blob.core.windows.net/container/checkpoint.ckpt",
                "loader_type": "transfer_learning",
            },
        ]

        for checkpoint_config in valid_configs:
            config_dict = {"training": {"checkpoint_loading": checkpoint_config}}

            # Should not raise validation error
            config = BaseSchema(**DictConfig(config_dict))
            assert config.training.checkpoint_loading is not None

    def test_checkpoint_loading_dict_type(self):
        """Test that checkpoint_loading accepts dict type."""
        config_dict = {
            "training": {
                "checkpoint_loading": {
                    "source": "/path/to/checkpoint.ckpt",
                    "loader_type": "weights_only",
                    "custom_param": "custom_value",  # Additional parameters should be allowed
                },
            },
        }

        config = BaseSchema(**DictConfig(config_dict))
        assert isinstance(config.training.checkpoint_loading, (dict, DictConfig))

    def test_checkpoint_loading_empty_dict(self):
        """Test that empty dict is valid for checkpoint_loading."""
        config_dict = {"training": {"checkpoint_loading": {}}}

        # Should not raise validation error - empty dict is valid
        config = BaseSchema(**DictConfig(config_dict))
        assert config.training.checkpoint_loading == {}

    def test_checkpoint_loading_none_explicit(self):
        """Test explicitly setting checkpoint_loading to None."""
        config_dict = {"training": {"checkpoint_loading": None}}

        config = BaseSchema(**DictConfig(config_dict))
        assert config.training.checkpoint_loading is None

    def test_checkpoint_loading_with_minimal_training_config(self):
        """Test checkpoint_loading with minimal training configuration."""
        # Create a minimal training config that would normally be valid
        config_dict = {
            "training": {
                "model_task": "anemoi.training.train.forecaster.GraphForecaster",
                "checkpoint_loading": {"source": "/path/to/checkpoint.ckpt", "loader_type": "weights_only"},
            },
        }

        # This might not fully validate due to missing required fields,
        # but checkpoint_loading field itself should be parsed correctly
        try:
            config = BaseSchema(**DictConfig(config_dict))
            assert config.training.checkpoint_loading is not None
        except ValidationError as e:
            # If validation fails, it shouldn't be due to checkpoint_loading field
            error_str = str(e)
            assert "checkpoint_loading" not in error_str

    def test_checkpoint_loading_with_complex_config(self):
        """Test checkpoint_loading works with complex nested configuration."""
        config_dict = {
            "training": {
                "checkpoint_loading": {
                    "source": "s3://my-bucket/experiments/run-123/checkpoint-best.ckpt",
                    "loader_type": "transfer_learning",
                    "strict": False,
                    "skip_mismatched": True,
                    "additional_options": {"nested": "value", "list": [1, 2, 3]},
                },
            },
        }

        config = BaseSchema(**DictConfig(config_dict))
        checkpoint_config = config.training.checkpoint_loading

        assert checkpoint_config["source"] == "s3://my-bucket/experiments/run-123/checkpoint-best.ckpt"
        assert checkpoint_config["loader_type"] == "transfer_learning"
        assert checkpoint_config["strict"] is False
        assert checkpoint_config["skip_mismatched"] is True
        assert "additional_options" in checkpoint_config

    def test_checkpoint_loading_field_description(self):
        """Test that the field has proper description/documentation."""
        from anemoi.training.schemas.training import BaseTrainingSchema

        # Check that the field exists in the schema
        fields = BaseTrainingSchema.model_fields
        assert "checkpoint_loading" in fields

        # Check field configuration
        field_info = fields["checkpoint_loading"]
        assert field_info.default is None  # Should default to None

        # Field should accept Union[dict, None]
        # The actual validation depends on pydantic implementation details
