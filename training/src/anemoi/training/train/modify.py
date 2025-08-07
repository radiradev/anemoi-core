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

    def __init__(self, submodules_to_freeze: DictConfig) -> None:
        """Initialize the freezing model modifier with a configuration.

        Parameters
        ----------
        submodules_to_freeze : DictConfig
            Configuration containing submodules to freeze.
        """
        self.submodules_to_freeze = submodules_to_freeze

    def apply(self, model: torch.nn.Module, config: DictConfig) -> torch.nn.Module:
        LOGGER.info("The following submodules will NOT be trained: %s", self.submodules_to_freeze)
        for module_name in self.submodules_to_freeze:
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


class TransferLearningModelModifier(ModelModifier):
    """Modifier for applying transfer learning from checkpoint."""

    def __init__(self, checkpoint_path: Path | str, strict: bool = False, skip_mismatched: bool = True) -> None:
        """Initialize the transfer learning model modifier with a checkpoint path.

        Parameters
        ----------
        checkpoint_path : str | Path
            The path to the checkpoint file for transfer learning.
        strict : bool, default False
            Whether to strictly enforce that the keys in state_dict match
        skip_mismatched : bool, default True
            Whether to skip parameters with shape mismatches
        """
        self.checkpoint_path = checkpoint_path
        self.strict = strict
        self.skip_mismatched = skip_mismatched

    def apply(self, model: torch.nn.Module, config: DictConfig) -> torch.nn.Module:
        """Load weights from checkpoint with size mismatch handling.

        Parameters
        ----------
        model : torch.nn.Module
            The model to load weights into
        config : DictConfig
            Configuration object

        Returns
        -------
        torch.nn.Module
            Model with loaded weights
        """
        from anemoi.training.utils.model_loading import load_model_from_checkpoint

        # Use #458's transfer learning loader
        model = load_model_from_checkpoint(
            model=model,
            checkpoint_source=self.checkpoint_path,
            loader_type="transfer_learning",
            strict=self.strict,
            skip_mismatched=self.skip_mismatched,
        )

        return model


class ModelModifierApplier:
    """Applies model modifiers to instantiated model."""

    def process(self, base_model: torch.nn.Module, config: DictConfig) -> torch.nn.Module:
        """Apply enabled modifiers in correct order."""
        model = base_model

        # Check if model_modifier config exists and has modifiers
        if not hasattr(config.training, "model_modifier") or not hasattr(config.training.model_modifier, "modifiers"):
            LOGGER.info("No model modifiers configured, returning model as-is")
            return model

        model_modifier_order = (
            instantiate(model_modifier) for model_modifier in config.training.model_modifier.modifiers
        )

        for model_modifier in model_modifier_order:
            LOGGER.info("Applying model modifier: %s", type(model_modifier).__name__)
            model = model_modifier.apply(model, config)

        return model
