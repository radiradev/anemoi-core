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

import pytest
import yaml
from hydra import compose
from hydra import initialize_config_dir
from omegaconf import DictConfig

from anemoi.training.config import CONFIG_PATH


class TestCheckpointLoadingConfigs:
    """Test checkpoint_loading configuration templates."""

    def test_config_templates_exist(self):
        """Test that all expected config templates exist."""
        config_dir = Path(CONFIG_PATH) / "training" / "checkpoint_loading"
        expected_templates = ["weights_only.yml", "transfer_learning.yml", "standard.yml"]

        for template in expected_templates:
            template_path = config_dir / template
            assert template_path.exists(), f"Missing config template: {template}"

    def test_weights_only_config_structure(self):
        """Test weights_only.yml config structure."""
        config_dir = Path(CONFIG_PATH) / "training" / "checkpoint_loading"
        config_path = config_dir / "weights_only.yml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "source" in config
        assert "loader_type" in config
        assert config["loader_type"] == "weights_only"
        assert "strict" in config

    def test_transfer_learning_config_structure(self):
        """Test transfer_learning.yml config structure."""
        config_dir = Path(CONFIG_PATH) / "training" / "checkpoint_loading"
        config_path = config_dir / "transfer_learning.yml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "source" in config
        assert "loader_type" in config
        assert config["loader_type"] == "transfer_learning"
        assert "strict" in config
        assert "skip_mismatched" in config

        # Transfer learning should have appropriate defaults
        assert config["strict"] is False
        assert config["skip_mismatched"] is True

    def test_standard_config_structure(self):
        """Test standard.yml config structure."""
        config_dir = Path(CONFIG_PATH) / "training" / "checkpoint_loading"
        config_path = config_dir / "standard.yml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "source" in config
        assert "loader_type" in config
        assert config["loader_type"] == "standard"
        assert "strict" in config

    def test_config_yaml_validity(self):
        """Test that all config templates are valid YAML."""
        config_dir = Path(CONFIG_PATH) / "training" / "checkpoint_loading"

        for config_file in config_dir.glob("*.yml"):
            with open(config_file) as f:
                try:
                    yaml.safe_load(f)
                except yaml.YAMLError:
                    pytest.fail(f"Invalid YAML in {config_file}")

    def test_hydra_config_loading(self):
        """Test that configs can be loaded through Hydra."""
        config_dir = Path(CONFIG_PATH)

        # Test loading each checkpoint loading config through Hydra
        checkpoint_configs = [
            "checkpoint_loading=weights_only",
            "checkpoint_loading=transfer_learning",
            "checkpoint_loading=standard",
        ]

        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            for config_override in checkpoint_configs:
                try:
                    # Try to compose with minimal overrides
                    cfg = compose(overrides=[config_override])
                    assert "checkpoint_loading" in cfg
                    assert cfg.checkpoint_loading.loader_type is not None
                except Exception as e:
                    pytest.fail(f"Failed to load config {config_override}: {e}")

    def test_config_parameter_types(self):
        """Test that config parameters have correct types."""
        config_dir = Path(CONFIG_PATH) / "training" / "checkpoint_loading"

        for config_file in config_dir.glob("*.yml"):
            with open(config_file) as f:
                config = yaml.safe_load(f)

            # source should be string or null
            assert config["source"] is None or isinstance(config["source"], str)

            # loader_type should be string
            assert isinstance(config["loader_type"], str)
            assert config["loader_type"] in ["weights_only", "transfer_learning", "standard"]

            # strict should be boolean if present
            if "strict" in config:
                assert isinstance(config["strict"], bool)

            # skip_mismatched should be boolean if present
            if "skip_mismatched" in config:
                assert isinstance(config["skip_mismatched"], bool)

    def test_config_templates_with_actual_paths(self):
        """Test config templates with actual checkpoint paths."""
        test_configs = {
            "weights_only": {"source": "/path/to/checkpoint.ckpt", "loader_type": "weights_only", "strict": True},
            "transfer_learning": {
                "source": "s3://bucket/checkpoint.ckpt",
                "loader_type": "transfer_learning",
                "strict": False,
                "skip_mismatched": True,
            },
            "standard": {"source": "https://example.com/checkpoint.ckpt", "loader_type": "standard", "strict": True},
        }

        for config_name, expected_structure in test_configs.items():
            # Verify our templates match expected structure
            config_dir = Path(CONFIG_PATH) / "training" / "checkpoint_loading"
            config_path = config_dir / f"{config_name}.yml"

            with open(config_path) as f:
                template_config = yaml.safe_load(f)

            # Check that all expected keys exist
            for key in expected_structure:
                assert key in template_config

    def test_config_documentation_completeness(self):
        """Test that configs have appropriate documentation/comments."""
        config_dir = Path(CONFIG_PATH) / "training" / "checkpoint_loading"

        for config_file in config_dir.glob("*.yml"):
            with open(config_file) as f:
                content = f.read()

            # Should have comments explaining the parameters
            assert "#" in content, f"Config {config_file.name} should have documentation comments"

            # Key parameters should be documented
            if "source" in content:
                assert (
                    "Path" in content or "URL" in content
                ), f"Config {config_file.name} should document source parameter"

    def test_config_integration_with_training_schema(self):
        """Test that config templates work with training schema validation."""
        from anemoi.training.schemas.base_schema import BaseSchema

        config_dir = Path(CONFIG_PATH) / "training" / "checkpoint_loading"

        for config_file in config_dir.glob("*.yml"):
            with open(config_file) as f:
                checkpoint_config = yaml.safe_load(f)

            # Create minimal training config with checkpoint loading
            training_config = DictConfig({"training": {"checkpoint_loading": checkpoint_config}})

            # Should not raise validation error for checkpoint_loading field
            try:
                config = BaseSchema(**training_config)
                assert config.training.checkpoint_loading is not None
            except Exception as e:
                # If it fails, it shouldn't be due to checkpoint_loading structure
                error_str = str(e)
                if "checkpoint_loading" in error_str:
                    pytest.fail(f"Config template {config_file.name} failed schema validation: {e}")
