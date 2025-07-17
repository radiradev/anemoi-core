# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import torch
from hydra.utils import instantiate

from anemoi.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from omegaconf import DictConfig

LOGGER = get_logger(__name__)


class ModelModifier(ABC):
    """Base model modifier to change loaded models post-init."""

    @abstractmethod
    def apply(self, model: torch.nn.Module, config: DictConfig) -> torch.nn.Module: ...


class FreezingModelModifier(ModelModifier):
    """Model modifier to freeze modules in a model."""

    def apply(self, model: torch.nn.Module, config: DictConfig) -> torch.nn.Module:
        LOGGER.info("The following submodules will NOT be trained: %s", self.config.submodules_to_freeze)
        for module_name in config.submodules_to_freeze:
            self._freeze_submodule_by_name(model, module_name)
            LOGGER.info("`%s` frozen successfully.", module_name)
        return model

    def _freeze_submodule_by_name(self, module: torch.nn.Module, target_name: str) -> None:
        """Recursively freezes the parameters of a submodule with the specified name.

        Parameters
        ----------
        module : torch.nn.Module
            Pytorch model
        target_name : str
            The name of the submodule to freeze.
        """
        for name, child in module.named_children():
            # If this is the target submodule, freeze its parameters
            if name == target_name:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                # Recursively search within children
                self._freeze_submodule_by_name(child, target_name)


class WeightsInitModelModifier(ModelModifier):
    """Modifier for initializing model weights."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize the weights initialization model modifier with the configuration.

        Parameters
        ----------
        config : DictConfig
            The configuration parameters for weight initialization.
        """
        self.config = config

    def apply(self, model: torch.nn.Module, config: DictConfig) -> torch.nn.Module:
        """Apply weight initialization to the model.

        Parameters
        ----------
        model : torch.nn.Module
            The model to initialize weights for
        config : DictConfig
            Configuration containing weight initialization parameters

        Returns
        -------
        torch.nn.Module
            Model with initialized weights
        """
        del config

        model.load_from_checkpoint(
            model.checkpoint_path,
            config=model.config,
            data_indices=model.data_indices,
            graph_data=model.graph_data,
            truncation_data=model.truncation_data,
            metadata=model.metadata,
            statistics=model.statistics,
            statistics_tendencies=model.statistics_tendencies,
            supporting_arrays=model.supporting_arrays,
            strict=False,
        )
        LOGGER.info("Restoring only model weights from %s", model.last_checkpoint)
        return model


class TransferLearningModelModifier(ModelModifier):
    """Modifier for applying transfer learning from checkpoint."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize the transfer learning model modifier with the configuration.

        Parameters
        ----------
        config : DictConfig
            The configuration parameters for transfer learning.
            Should contain 'ckpt_path' field.
        """
        self.config = config

    def apply(self, model: torch.nn.Module, config: DictConfig) -> torch.nn.Module:
        """Apply transfer learning by loading weights from checkpoint.

        Parameters
        ----------
        model : torch.nn.Module
            The model to load weights into
        config : DictConfig
            Configuration containing transfer learning parameters

        Returns
        -------
        torch.nn.Module
            Model with transferred weights loaded
        """
        # Get checkpoint path from model modifier config or main config
        ckpt_path = getattr(self.config, "ckpt_path", None)
        if ckpt_path is None:
            ckpt_path = getattr(config.finetuning.transfer_learning, "ckpt_path", None)

        if ckpt_path is None:
            err = "No checkpoint path specified for transfer learning"
            raise ValueError(err)

        return self._transfer_learning_loading(model, ckpt_path)

    def _transfer_learning_loading(self, model: torch.nn.Module, ckpt_path: Path | str) -> torch.nn.Module:
        """Load weights from checkpoint with size mismatch handling.

        Parameters
        ----------
        model : torch.nn.Module
            The model to load weights into
        ckpt_path : Path | str
            Path to the checkpoint file

        Returns
        -------
        torch.nn.Module
            Model with loaded weights
        """
        # Load the checkpoint
        checkpoint = torch.load(ckpt_path, weights_only=False, map_location=model.device)

        # Filter out layers with size mismatch
        state_dict = checkpoint["state_dict"]
        model_state_dict = model.state_dict()

        for key in state_dict.copy():
            if key in model_state_dict and state_dict[key].shape != model_state_dict[key].shape:
                LOGGER.info("Skipping loading parameter: %s", key)
                LOGGER.info("Checkpoint shape: %s", str(state_dict[key].shape))
                LOGGER.info("Model shape: %s", str(model_state_dict[key].shape))
                del state_dict[key]  # Remove the mismatched key

        # Load the filtered state_dict into the model
        model.load_state_dict(state_dict, strict=False)
        model.weights_initialized = True
        LOGGER.info("Transfer learning applied successfully from %s", ckpt_path)

        return model


class ModelModifierApplier:
    """Applies model modifiers to instantiated model."""

    def process(self, base_model: torch.nn.Module, config: DictConfig) -> torch.nn.Module:
        """Apply enabled modifiers in correct order."""
        model = base_model

        model_modifier_order = (
            instantiate(model_modifier) for model_modifier in config.finetuning.strategies
        )  # TODO(@JesperDramsch): find the right place in the config

        for model_modifier in model_modifier_order:
            model = model_modifier.apply(model, config)

        return model
