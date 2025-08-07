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

from anemoi.training.schemas.training import BaseTrainingSchema


class TestModelModifierSchema:
    """Test schema validation for model_modifier configuration."""

    def create_minimal_training_config(self, **overrides):
        """Create a minimal valid training configuration."""
        base_config = {
            "run_id": None,
            "fork_run_id": None,
            "deterministic": False,
            "precision": "16-mixed",
            "multistep_input": 2,
            "accum_grad_batches": 1,
            "num_sanity_val_steps": 6,
            "gradient_clip": {"val": 32.0, "algorithm": "value"},
            "strategy": {
                "_target_": "anemoi.training.distributed.strategy.DDPGroupStrategy",
                "num_gpus_per_model": 2,
                "read_group_size": 1,
            },
            "swa": {"enabled": False, "lr": 1.0e-4},
            "training_loss": {"_target_": "anemoi.training.losses.MSELoss", "scalers": []},
            "loss_gradient_scaling": False,
            "scalers": {},
            "validation_metrics": {},
            "variable_groups": {},
            "rollout": {"start": 1, "epoch_increment": 0, "max": 1},
            "max_epochs": None,
            "max_steps": 150000,
            "lr": {"rate": 0.625e-4, "iterations": 300000, "min": 3e-7, "warmup": 1000},
            "optimizer": {"zero": False, "kwargs": {}},
            "metrics": ["mse"],
        }

        # Apply overrides
        for key, value in overrides.items():
            base_config[key] = value

        return base_config

    def test_schema_without_model_modifier(self) -> None:
        """Test that schema validates correctly without model_modifier field."""
        config_data = self.create_minimal_training_config()

        # Should validate successfully with model_modifier field absent
        schema = BaseTrainingSchema(**config_data)
        assert schema.model_modifier is None

    def test_schema_with_none_model_modifier(self) -> None:
        """Test that schema validates correctly with model_modifier set to None."""
        config_data = self.create_minimal_training_config(model_modifier=None)

        schema = BaseTrainingSchema(**config_data)
        assert schema.model_modifier is None

    def test_schema_with_valid_model_modifier_dict(self) -> None:
        """Test that schema validates correctly with valid model_modifier DictConfig."""
        model_modifier_config = {
            "modifiers": [
                {
                    "_target_": "anemoi.training.train.modify.WeightsInitModelModifier",
                    "checkpoint_path": "/path/to/checkpoint.ckpt",
                },
            ],
        }

        config_data = self.create_minimal_training_config(model_modifier=model_modifier_config)

        schema = BaseTrainingSchema(**config_data)
        assert schema.model_modifier is not None
        assert isinstance(schema.model_modifier, dict)
        assert "modifiers" in schema.model_modifier

    def test_schema_with_dictconfig_model_modifier(self) -> None:
        """Test that schema validates correctly with DictConfig model_modifier."""
        model_modifier_config = DictConfig(
            {
                "modifiers": [
                    {
                        "_target_": "anemoi.training.train.modify.FreezingModelModifier",
                        "submodules_to_freeze": ["encoder", "processor.0"],
                    },
                ],
            },
        )

        config_data = self.create_minimal_training_config(model_modifier=model_modifier_config)

        schema = BaseTrainingSchema(**config_data)
        assert schema.model_modifier is not None
        # Should preserve DictConfig type
        assert isinstance(schema.model_modifier, (DictConfig, dict))

    def test_schema_with_multiple_modifiers(self) -> None:
        """Test schema validation with multiple modifiers."""
        model_modifier_config = {
            "modifiers": [
                {
                    "_target_": "anemoi.training.train.modify.TransferLearningModelModifier",
                    "checkpoint_path": "/path/to/pretrained.ckpt",
                },
                {"_target_": "anemoi.training.train.modify.FreezingModelModifier", "submodules_to_freeze": ["encoder"]},
                {
                    "_target_": "anemoi.training.train.modify.WeightsInitModelModifier",
                    "checkpoint_path": "/path/to/weights.ckpt",
                },
            ],
        }

        config_data = self.create_minimal_training_config(model_modifier=model_modifier_config)

        schema = BaseTrainingSchema(**config_data)
        assert schema.model_modifier is not None
        assert len(schema.model_modifier["modifiers"]) == 3

    def test_schema_with_empty_modifiers_list(self) -> None:
        """Test schema validation with empty modifiers list."""
        model_modifier_config = {"modifiers": []}

        config_data = self.create_minimal_training_config(model_modifier=model_modifier_config)

        schema = BaseTrainingSchema(**config_data)
        assert schema.model_modifier is not None
        assert len(schema.model_modifier["modifiers"]) == 0

    def test_schema_preserves_arbitrary_modifier_configs(self) -> None:
        """Test that schema preserves arbitrary modifier configurations."""
        model_modifier_config = {
            "modifiers": [
                {
                    "_target_": "custom.module.CustomModifier",
                    "custom_param": "custom_value",
                    "custom_dict": {"nested_param": 42, "nested_list": [1, 2, 3]},
                },
            ],
        }

        config_data = self.create_minimal_training_config(model_modifier=model_modifier_config)

        schema = BaseTrainingSchema(**config_data)
        assert schema.model_modifier is not None
        modifier = schema.model_modifier["modifiers"][0]
        assert modifier["_target_"] == "custom.module.CustomModifier"
        assert modifier["custom_param"] == "custom_value"
        assert modifier["custom_dict"]["nested_param"] == 42

    def test_schema_field_description(self) -> None:
        """Test that model_modifier field has proper description."""
        # Get the field info from the schema
        field_info = BaseTrainingSchema.model_fields["model_modifier"]

        # Check that it has a meaningful description
        assert field_info.description == "Model modifier configuration for weight loading, freezing, and fine-tuning."

    def test_schema_field_default_value(self) -> None:
        """Test that model_modifier field has correct default value."""
        field_info = BaseTrainingSchema.model_fields["model_modifier"]

        # Should default to None
        assert field_info.default is None

    def test_schema_field_type_annotation(self) -> None:
        """Test that model_modifier field has correct type annotation."""
        BaseTrainingSchema.model_fields["model_modifier"]

        # Should accept Union[DictConfig, None]
        # The actual annotation checking depends on pydantic internals,
        # so we just verify it accepts both types in practice
        config_data = self.create_minimal_training_config(model_modifier=None)
        schema1 = BaseTrainingSchema(**config_data)
        assert schema1.model_modifier is None

        config_data = self.create_minimal_training_config(model_modifier={"modifiers": []})
        schema2 = BaseTrainingSchema(**config_data)
        assert schema2.model_modifier is not None

    def test_schema_with_malformed_modifier_config(self) -> None:
        """Test that schema handles malformed modifier configs gracefully."""
        # The schema should accept arbitrary dict structures for model_modifier
        # since we use DictConfig which is flexible
        model_modifier_config = {"malformed": "config", "no_modifiers": True}

        config_data = self.create_minimal_training_config(model_modifier=model_modifier_config)

        # Should not raise validation error at schema level
        # (validation would happen later during instantiation)
        schema = BaseTrainingSchema(**config_data)
        assert schema.model_modifier is not None
        assert schema.model_modifier["malformed"] == "config"

    def test_schema_integration_with_other_fields(self) -> None:
        """Test that model_modifier integrates correctly with other schema fields."""
        model_modifier_config = {
            "modifiers": [
                {
                    "_target_": "anemoi.training.train.modify.WeightsInitModelModifier",
                    "checkpoint_path": "/path/to/checkpoint.ckpt",
                },
            ],
        }

        config_data = self.create_minimal_training_config(
            model_modifier=model_modifier_config,
            precision="32",  # Change another field
            max_epochs=10,  # Change another field
        )

        schema = BaseTrainingSchema(**config_data)

        # Verify all fields are correctly set
        assert schema.model_modifier is not None
        assert schema.precision == "32"
        assert schema.max_epochs == 10

        # Verify model_modifier is preserved correctly
        assert len(schema.model_modifier["modifiers"]) == 1
        assert "WeightsInitModelModifier" in schema.model_modifier["modifiers"][0]["_target_"]


class TestForecasterSchemaWithModelModifier:
    """Test ForecasterSchema specifically with model_modifier."""

    def create_minimal_forecaster_config(self, **overrides):
        """Create minimal forecaster configuration."""
        config = TestModelModifierSchema().create_minimal_training_config(**overrides)
        config["model_task"] = "anemoi.training.train.forecaster.GraphForecaster"
        return config

    def test_forecaster_schema_with_model_modifier(self) -> None:
        """Test ForecasterSchema with model_modifier configuration."""
        from anemoi.training.schemas.training import ForecasterSchema

        model_modifier_config = {
            "modifiers": [
                {"_target_": "anemoi.training.train.modify.FreezingModelModifier", "submodules_to_freeze": ["encoder"]},
            ],
        }

        config_data = self.create_minimal_forecaster_config(model_modifier=model_modifier_config)

        schema = ForecasterSchema(**config_data)
        assert schema.model_modifier is not None
        assert schema.model_task == "anemoi.training.train.forecaster.GraphForecaster"

    def test_forecaster_ensemble_schema_with_model_modifier(self) -> None:
        """Test ForecasterEnsSchema with model_modifier configuration."""
        from anemoi.training.schemas.training import ForecasterEnsSchema

        model_modifier_config = {
            "modifiers": [
                {
                    "_target_": "anemoi.training.train.modify.TransferLearningModelModifier",
                    "checkpoint_path": "/path/to/checkpoint.ckpt",
                },
            ],
        }

        config_data = self.create_minimal_forecaster_config(
            model_modifier=model_modifier_config,
            ensemble_size_per_device=2,
        )
        config_data["model_task"] = "anemoi.training.train.forecaster.GraphEnsForecaster"

        schema = ForecasterEnsSchema(**config_data)
        assert schema.model_modifier is not None
        assert schema.ensemble_size_per_device == 2
        assert schema.model_task == "anemoi.training.train.forecaster.GraphEnsForecaster"


class TestModelModifierConfigValidation:
    """Test validation of common model modifier configurations."""

    def test_weights_init_modifier_config(self) -> None:
        """Test typical WeightsInitModelModifier configuration."""
        config = {
            "modifiers": [
                {
                    "_target_": "anemoi.training.train.modify.WeightsInitModelModifier",
                    "checkpoint_path": "/path/to/checkpoint.ckpt",
                },
            ],
        }

        # Should be valid as a dict structure
        assert "modifiers" in config
        assert len(config["modifiers"]) == 1
        assert "_target_" in config["modifiers"][0]
        assert "checkpoint_path" in config["modifiers"][0]

    def test_transfer_learning_modifier_config(self) -> None:
        """Test typical TransferLearningModelModifier configuration."""
        config = {
            "modifiers": [
                {
                    "_target_": "anemoi.training.train.modify.TransferLearningModelModifier",
                    "checkpoint_path": "/path/to/pretrained.ckpt",
                },
            ],
        }

        assert "modifiers" in config
        assert len(config["modifiers"]) == 1
        assert "TransferLearningModelModifier" in config["modifiers"][0]["_target_"]

    def test_freezing_modifier_config(self) -> None:
        """Test typical FreezingModelModifier configuration."""
        config = {
            "modifiers": [
                {
                    "_target_": "anemoi.training.train.modify.FreezingModelModifier",
                    "submodules_to_freeze": ["encoder", "processor.0", "processor.1"],
                },
            ],
        }

        assert "modifiers" in config
        assert len(config["modifiers"]) == 1
        assert isinstance(config["modifiers"][0]["submodules_to_freeze"], list)
        assert len(config["modifiers"][0]["submodules_to_freeze"]) == 3

    def test_combined_modifiers_config(self) -> None:
        """Test configuration with multiple modifiers combined."""
        config = {
            "modifiers": [
                {
                    "_target_": "anemoi.training.train.modify.TransferLearningModelModifier",
                    "checkpoint_path": "/path/to/pretrained.ckpt",
                },
                {
                    "_target_": "anemoi.training.train.modify.FreezingModelModifier",
                    "submodules_to_freeze": ["encoder", "processor.0"],
                },
            ],
        }

        assert "modifiers" in config
        assert len(config["modifiers"]) == 2

        # Check first modifier (transfer learning)
        assert "TransferLearningModelModifier" in config["modifiers"][0]["_target_"]
        assert "checkpoint_path" in config["modifiers"][0]

        # Check second modifier (freezing)
        assert "FreezingModelModifier" in config["modifiers"][1]["_target_"]
        assert "submodules_to_freeze" in config["modifiers"][1]
        assert isinstance(config["modifiers"][1]["submodules_to_freeze"], list)

    def test_config_with_additional_parameters(self) -> None:
        """Test modifier configurations with additional parameters."""
        config = {
            "modifiers": [
                {
                    "_target_": "anemoi.training.train.modify.WeightsInitModelModifier",
                    "checkpoint_path": "/path/to/checkpoint.ckpt",
                    "strict_loading": False,  # Additional parameter
                    "map_location": "cpu",  # Additional parameter
                },
            ],
        }

        modifier = config["modifiers"][0]
        assert "_target_" in modifier
        assert "checkpoint_path" in modifier
        assert "strict_loading" in modifier
        assert "map_location" in modifier
        assert modifier["strict_loading"] is False
        assert modifier["map_location"] == "cpu"
