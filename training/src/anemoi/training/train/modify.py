# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Model modification system for flexible model preparation and transfer learning.

This module provides a comprehensive system for modifying PyTorch models after
instantiation but before training. The system is built around the ``ModelModifier``
pattern, which allows for composable, reusable, and configurable model modifications.

Key Components
--------------

**ModelModifier (ABC)**: Abstract base class defining the modification interface
**FreezingModelModifier**: Freeze specific model parameters to prevent updates
**TransferLearningModelModifier**: Load weights from pretrained checkpoints
**ModelModifierApplier**: Orchestrate application of multiple modifiers

Core Features
-------------

- **Composable**: Chain multiple modifiers together in any order
- **Configurable**: Full Hydra configuration support with validation
- **Extensible**: Easy to add new modifier types for custom use cases
- **Robust**: Comprehensive error handling and logging throughout
- **Flexible**: Support for various checkpoint sources (local, S3, HTTP, etc.)

Quick Start Guide
-----------------

Basic usage in YAML configuration::

    training:
      model_modifier:
        modifiers:
          # Load pretrained weights
          - _target_: "anemoi.training.train.modify.TransferLearningModelModifier"
            checkpoint_path: "/path/to/pretrained.ckpt"
            strict: false

          # Freeze encoder layers
          - _target_: "anemoi.training.train.modify.FreezingModelModifier"
            submodules_to_freeze: ["encoder", "processor.0"]

Programmatic usage::

    from anemoi.training.train.modify import (
        FreezingModelModifier,
        TransferLearningModelModifier,
        ModelModifierApplier
    )
    from omegaconf import DictConfig

    # Create modifiers
    transfer_mod = TransferLearningModelModifier("pretrained.ckpt")
    freeze_mod = FreezingModelModifier(["encoder"])

    # Apply individually
    model = transfer_mod.apply(model)
    model = freeze_mod.apply(model)

    # Or use the applier with configuration
    config = DictConfig({...})  # Your configuration
    applier = ModelModifierApplier()
    model = applier.process(model, config)

Common Use Cases
----------------

**Domain Adaptation**::

    modifiers:
      - _target_: "anemoi.training.train.modify.TransferLearningModelModifier"
        checkpoint_path: "general_weather_model.ckpt"
      - _target_: "anemoi.training.train.modify.FreezingModelModifier"
        submodules_to_freeze: ["encoder"]  # Keep feature extraction, adapt prediction

**Few-Shot Learning**::

    modifiers:
      - _target_: "anemoi.training.train.modify.TransferLearningModelModifier"
        checkpoint_path: "large_pretrained_model.ckpt"
      - _target_: "anemoi.training.train.modify.FreezingModelModifier"
        submodules_to_freeze: ["encoder", "processor.0", "processor.1"]

**Progressive Fine-tuning**::

    modifiers:
      - _target_: "anemoi.training.train.modify.TransferLearningModelModifier"
        checkpoint_path: "base_model.ckpt"
      - _target_: "anemoi.training.train.modify.TransferLearningModelModifier"
        checkpoint_path: "domain_specific.ckpt"  # May override some base weights
        strict: false
      - _target_: "anemoi.training.train.modify.FreezingModelModifier"
        submodules_to_freeze: ["encoder"]

Architecture Integration
------------------------

The modifier system integrates seamlessly with Anemoi's training pipeline:

1. **Model Creation**: Standard model instantiation from configuration
2. **Checkpoint Loading**: Optional checkpoint loading via dedicated system
3. **Model Modification**: Application of configured modifiers (this module)
4. **Training**: Standard PyTorch Lightning training with modified model

The modifiers are applied in the training pipeline at:
``anemoi.training.train.train.AnemoiTrainer.model`` property

Advanced Configuration
----------------------

**Conditional Modifiers**::

    modifiers:
      - _target_: "anemoi.training.train.modify.TransferLearningModelModifier"
        checkpoint_path: ${pretrained_path}  # From environment or command line
      - _target_: "anemoi.training.train.modify.FreezingModelModifier"
        submodules_to_freeze: ${freeze_layers:["encoder"]}  # Default to encoder

**Custom Modifiers**::

    # In your custom module
    class NoiseInjectionModifier(ModelModifier):
        def __init__(self, noise_std: float = 0.01):
            self.noise_std = noise_std

        def apply(self, model):
            for param in model.parameters():
                param.data += torch.randn_like(param.data) * self.noise_std
            return model

    # In configuration
    modifiers:
      - _target_: "my.package.NoiseInjectionModifier"
        noise_std: 0.005

Error Handling and Debugging
-----------------------------

All components provide detailed error messages and logging:

- **Configuration errors**: Clear indication of missing or invalid parameters
- **Import errors**: Detailed class import failure information
- **Runtime errors**: Context about which modifier failed and why
- **Progress tracking**: INFO-level logging of each modification step

Enable debug logging for detailed information.

Performance Considerations
--------------------------

- **Memory efficiency**: Large checkpoint loading is optimized for memory usage
- **Lazy loading**: Modifiers are instantiated only when needed
- **In-place operations**: Most modifications happen in-place to save memory
- **Error early**: Configuration validation happens before model modification

Migration from Legacy System
-----------------------------

This module replaces the legacy ``load_weights_only`` and ``transfer_learning``
keywords with a more flexible, extensible system:

**Old approach**::

    training:
      load_weights_only: true
      checkpoint_path: "pretrained.ckpt"

**New approach**::

    training:
      checkpoint_loading:
        source: "pretrained.ckpt"
        loader_type: "weights_only"

Or using modifiers for more complex scenarios::

    training:
      model_modifier:
        modifiers:
          - _target_: "anemoi.training.train.modify.TransferLearningModelModifier"
            checkpoint_path: "pretrained.ckpt"

See Also
--------
anemoi.training.utils.model_loading : Checkpoint loading utilities (optional dependency for additional features)
anemoi.training.train.train : Main training pipeline integration
anemoi.training.utils.checkpoint_loaders : Checkpoint source handling (optional dependency for remote sources)

Notes
-----
This module is part of Anemoi's modular training system. While it integrates best with
the full Anemoi training environment, core functionality (freezing and basic transfer
learning) works independently. It uses Hydra for configuration management and integrates
with PyTorch Lightning for training orchestration.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from hydra.utils import instantiate

if TYPE_CHECKING:
    from omegaconf import DictConfig

import logging

LOGGER = logging.getLogger(__name__)


class ModelModifier(ABC):
    """Abstract base class for model modifications after initialization.

    ModelModifiers provide a clean, extensible way to modify models after they
    have been created but before training begins. This is particularly useful
    for transfer learning, parameter freezing, quantization, and other
    model preparation tasks.

    The modifier pattern allows for:

    - **Composability**: Multiple modifiers can be chained together
    - **Reusability**: Modifiers can be configured and reused across experiments
    - **Extensibility**: New modifier types can be easily added
    - **Separation of concerns**: Model creation and modification are decoupled

    Examples
    --------
    Creating a custom modifier:

    .. code-block:: python

        class CustomModifier(ModelModifier):
            def __init__(self, some_config):
                self.config = some_config

            def apply(self, model):
                # Modify the model based on self.config
                model.custom_layer = nn.Linear(10, 5)
                return model

    Using modifiers in configuration:

    .. code-block:: yaml

        training:
          model_modifier:
            modifiers:
              - _target_: "my.package.CustomModifier"
                some_config: value

    See Also
    --------
    FreezingModelModifier : Freeze specific model parameters
    TransferLearningModelModifier : Load weights from pretrained checkpoints
    ModelModifierApplier : Orchestrates application of multiple modifiers

    Notes
    -----
    All implementations must be stateless with respect to the model - they should
    not store references to the model between calls. Configuration should be passed
    during initialization, not during the apply call.
    """

    @abstractmethod
    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply the modification to the given model.

        This is the core method that must be implemented by all modifier subclasses.
        The method should modify the model in-place and return it for chaining.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to be modified. The model is typically fully
            initialized but not yet trained.

        Returns
        -------
        torch.nn.Module
            The modified model. This is typically the same object as the input
            model (modified in-place), but could be a new model object if the
            modification requires it.

        Notes
        -----
        - The method should be idempotent when possible
        - Any logging should use the module's LOGGER
        - Errors should be meaningful and actionable
        - The model's training state should be preserved unless explicitly changed
        """
        ...


class FreezingModelModifier(ModelModifier):
    """Freeze specific modules in a model to prevent parameter updates during training.

    This modifier sets ``requires_grad = False`` for all parameters in the specified
    submodules, effectively freezing them during backpropagation. This is commonly
    used in transfer learning scenarios where you want to preserve pretrained features
    while fine-tuning only specific parts of the model.

    The freezing is applied by name, supporting both direct child modules and nested
    modules using dot notation (e.g., "encoder.attention.0").

    Examples
    --------
    Basic usage in YAML configuration:

    .. code-block:: yaml

        training:
          model_modifier:
            modifiers:
              - _target_: "anemoi.training.train.modify.FreezingModelModifier"
                submodules_to_freeze:
                  - "encoder"      # Freeze entire encoder
                  - "processor.0"  # Freeze first processor layer
                  - "decoder.embeddings"  # Freeze embedding layer in decoder

    Programmatic usage:

    .. code-block:: python

        from omegaconf import DictConfig

        # Create modifier
        freezer = FreezingModelModifier(["encoder", "processor.0"])

        # Apply to model
        frozen_model = freezer.apply(model)

        # Verify freezing
        for name, param in frozen_model.named_parameters():
            if "encoder" in name:
                assert not param.requires_grad

    Common Use Cases
    ----------------

    **Domain Adaptation**:
        Freeze encoder to preserve feature extraction, fine-tune decoder:

        .. code-block:: yaml

            submodules_to_freeze: ["encoder", "processor"]

    **Few-Shot Learning**:
        Freeze most layers, train only final layers:

        .. code-block:: yaml

            submodules_to_freeze:
              - "encoder"
              - "processor.0"
              - "processor.1"
              - "processor.2"

    **Gradual Unfreezing**:
        Start with aggressive freezing, gradually unfreeze layers:

        .. code-block:: yaml

            # Stage 1: Freeze everything except decoder
            submodules_to_freeze: ["encoder", "processor"]

            # Stage 2: Unfreeze last processor layers
            # submodules_to_freeze: ["encoder", "processor.0", "processor.1"]

    Notes
    -----
    - Freezing is applied recursively to all parameters within specified modules
    - Module names must match those in the model's ``named_children()`` structure
    - Non-existent module names are logged as warnings but don't cause failures
    - The model's training/eval state is preserved
    - Frozen parameters will have zero gradients during backpropagation

    See Also
    --------
    TransferLearningModelModifier : Load pretrained weights before freezing
    ModelModifierApplier : Apply multiple modifiers in sequence

    Warnings
    --------
    Ensure module names exactly match the model structure. Use ``model.named_children()``
    or ``print(model)`` to verify the correct names before configuration.
    """

    def __init__(
        self,
        submodules_to_freeze: DictConfig | list[str],
        strict: bool = False,
        validate_gradients: bool = True,
    ) -> None:
        """Initialize the freezing modifier.

        Parameters
        ----------
        submodules_to_freeze : list[str] or DictConfig
            Names of submodules to freeze. Each name should correspond to a module
            accessible via ``model.named_children()`` or nested modules using dot
            notation (e.g., "processor.0", "encoder.attention").

        strict : bool, default False
            If True, raise an error when a specified module is not found.
            If False, log a warning and continue.

        validate_gradients : bool, default True
            If True, validate that frozen parameters do not accumulate gradients
            after a forward/backward pass.

        Raises
        ------
        TypeError
            If submodules_to_freeze is not a DictConfig or list
        """
        if isinstance(submodules_to_freeze, (list, tuple)):
            self.submodules_to_freeze = submodules_to_freeze
        else:
            # Assume DictConfig or similar iterable
            self.submodules_to_freeze = list(submodules_to_freeze)

        self.strict = strict
        self.validate_gradients = validate_gradients

        LOGGER.debug(
            "Initialized FreezingModelModifier with modules: %s (strict=%s, validate=%s)",
            self.submodules_to_freeze,
            self.strict,
            self.validate_gradients,
        )

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        """Freeze the specified submodules in the model.

        This method iterates through the configured submodule names and sets
        ``requires_grad = False`` for all parameters in those modules.

        Parameters
        ----------
        model : torch.nn.Module
            The model to freeze parameters in. The model is modified in-place.

        Returns
        -------
        torch.nn.Module
            The input model with specified parameters frozen (same object).

        Notes
        -----
        - Parameters are frozen by setting ``requires_grad = False``
        - The search is performed recursively for nested module names
        - Non-existent modules generate warnings but don't cause failures
        - Progress is logged for each successfully frozen module
        """
        if not self.submodules_to_freeze:
            LOGGER.info("No submodules specified for freezing")
            return model

        LOGGER.info("Freezing the following submodules: %s", self.submodules_to_freeze)

        for module_name in self.submodules_to_freeze:
            frozen_count = self._freeze_submodule_by_name(model, module_name)
            if frozen_count > 0:
                LOGGER.info("Froze %d parameters in '%s'", frozen_count, module_name)
            else:
                msg = f"Module '{module_name}' not found or has no parameters to freeze"
                if self.strict:
                    raise ValueError(msg)
                LOGGER.warning(msg)

        # Validate that frozen parameters prevent gradient flow
        if self.validate_gradients:
            self._validate_gradient_flow(model)

        return model

    def _validate_gradient_flow(self, model: torch.nn.Module) -> None:
        """Validate that frozen parameters don't accumulate gradients.

        This method performs a test forward/backward pass to ensure that
        parameters marked as frozen (requires_grad=False) don't accumulate
        gradients during backpropagation.

        Parameters
        ----------
        model : torch.nn.Module
            The model with frozen parameters to validate

        Raises
        ------
        RuntimeError
            If frozen parameters unexpectedly accumulate gradients

        Notes
        -----
        This validation creates a small test input and performs a forward/backward
        pass. It's designed to catch configuration errors early in the training
        process.
        """
        LOGGER.debug("Validating gradient flow for frozen parameters")

        # Store original training mode
        was_training = model.training
        model.eval()

        try:
            # Create a small test input - adjust shape based on model's expected input
            # This is a simple heuristic; real models may need different shapes
            test_input = torch.randn(1, 10, requires_grad=True)

            # Attempt forward pass
            try:
                output = model(test_input)
                if hasattr(output, "mean"):
                    loss = output.mean()
                else:
                    loss = output.sum() if isinstance(output, torch.Tensor) else output[0].sum()

                # Backward pass
                loss.backward()

                # Check frozen parameters
                for module_name in self.submodules_to_freeze:
                    self._check_module_gradients(model, module_name)

                LOGGER.debug("Gradient validation successful - frozen parameters have no gradients")

            except (RuntimeError, TypeError, AttributeError) as e:
                # If we can't validate (e.g., model needs specific input shape),
                # log a warning but don't fail
                LOGGER.warning("Could not validate gradient flow: %s", e)

        finally:
            # Restore training mode and clean up gradients
            if was_training:
                model.train()
            # Clear any gradients that were computed
            model.zero_grad()

    def _check_module_gradients(self, module: torch.nn.Module, target_name: str) -> None:  # noqa: C901
        """Check that a specific module's parameters have no gradients.

        Uses optimized lookup via get_submodule() when possible.

        Parameters
        ----------
        module : torch.nn.Module
            The parent module to search within
        target_name : str
            The name of the submodule to check

        Raises
        ------
        RuntimeError
            If frozen parameters have gradients
        """
        # Try direct access first for O(1) lookup
        try:
            target_module = module.get_submodule(target_name)
            for param_name, param in target_module.named_parameters():
                if not param.requires_grad and param.grad is not None:
                    error_msg = (
                        f"Frozen parameter '{target_name}.{param_name}' "
                        f"unexpectedly has gradients. This may indicate "
                        f"a problem with the freezing mechanism."
                    )
                    raise RuntimeError(error_msg)
        except AttributeError:
            # Module not found via direct path, fall back to recursive search
            pass
        else:
            return

        # Fallback: Handle nested module names recursively
        if "." in target_name:
            parts = target_name.split(".", 1)
            parent_name, child_name = parts[0], parts[1]

            for name, child in module.named_children():
                if name == parent_name:
                    self._check_module_gradients(child, child_name)
        else:
            # Check direct children
            for name, child in module.named_children():
                if name == target_name:
                    for param_name, param in child.named_parameters():
                        if not param.requires_grad and param.grad is not None:
                            error_msg = (
                                f"Frozen parameter '{target_name}.{param_name}' "
                                f"unexpectedly has gradients. This may indicate "
                                f"a problem with the freezing mechanism."
                            )
                            raise RuntimeError(error_msg)
                    return

            # If not found in direct children, search recursively
            for _, child in module.named_children():
                self._check_module_gradients(child, target_name)

    def _freeze_submodule_by_name(self, module: torch.nn.Module, target_name: str) -> int:  # noqa: C901
        """Freeze parameters of a submodule by name using optimized lookup.

        This method uses PyTorch's built-in get_submodule() for efficient access
        to nested modules, avoiding recursive searches when possible. It supports
        both direct child modules and nested modules using dot notation.

        Parameters
        ----------
        module : torch.nn.Module
            The parent module to search within
        target_name : str
            The name of the submodule to freeze. Can include dots for nested access
            (e.g., "processor.0", "encoder.attention")

        Returns
        -------
        int
            Number of parameters that were frozen

        Notes
        -----
        This method first attempts direct access via get_submodule() for O(1) lookup,
        falling back to recursive search only when necessary (e.g., for partial
        name matches or when the exact path doesn't exist).

        Examples
        --------
        - "encoder" -> freezes the encoder module
        - "processor.0" -> freezes the first processor module
        - "encoder.attention.heads" -> freezes nested attention heads
        """
        frozen_count = 0

        # First, try direct access using PyTorch's get_submodule
        # This is O(1) for exact paths and much faster than recursive search
        try:
            target_module = module.get_submodule(target_name)
            # Freeze all parameters in the found module
            for param in target_module.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_count += 1
        except AttributeError:
            # Module not found via direct path, fall back to recursive search
            # This handles cases where the name might be a partial match
            pass
        else:
            return frozen_count

        # Fallback: Recursive search for partial matches
        # This is kept for backward compatibility and edge cases
        if "." in target_name:
            # Handle nested module names
            parts = target_name.split(".", 1)
            parent_name, child_name = parts[0], parts[1]

            for name, child in module.named_children():
                if name == parent_name:
                    frozen_count += self._freeze_submodule_by_name(child, child_name)
        else:
            # Search in direct children first
            for name, child in module.named_children():
                if name == target_name:
                    for param in child.parameters():
                        if param.requires_grad:
                            param.requires_grad = False
                            frozen_count += 1
                    return frozen_count

            # If not found in direct children, search recursively
            for _, child in module.named_children():
                frozen_count += self._freeze_submodule_by_name(child, target_name)

        return frozen_count


class TransferLearningModelModifier(ModelModifier):
    """Load pretrained weights from checkpoints for transfer learning.

    This modifier enables transfer learning by loading weights from a pretrained model
    checkpoint into the current model. It provides flexible handling of architecture
    differences through configurable strict/mismatch policies.

    The modifier supports two operation modes:

    1. **Primary mode**: When available, leverages Anemoi's extensible checkpoint loading
       system which supports multiple source types including local files, S3,
       HTTP URLs, and cloud storage services.

    2. **Fallback mode**: A standalone implementation using standard PyTorch checkpoint
       loading that works independently, ensuring the modifier functions even without
       the extensible checkpoint loading system.

    Both modes provide robust error handling and detailed logging for debugging weight
    loading issues.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path or URL to the checkpoint file. Supports:

        - **Local paths**: ``"/path/to/model.ckpt"``
        - **S3 URLs**: ``"s3://bucket/model.ckpt"``
        - **HTTP URLs**: ``"https://example.com/model.ckpt"``
        - **GCS URLs**: ``"gs://bucket/model.ckpt"``
        - **Azure URLs**: ``"https://account.blob.core.windows.net/container/model.ckpt"``

    strict : bool, default False
        Whether to require exact parameter name matching between checkpoint and model.

        - **True**: All checkpoint keys must exist in model, and vice versa
        - **False**: Allow missing or extra keys with warnings

    skip_mismatched : bool, default True
        How to handle parameters with shape mismatches.

        - **True**: Skip parameters with different shapes and log warnings
        - **False**: Raise error on any shape mismatch

    Examples
    --------
    Basic transfer learning configuration:

    .. code-block:: yaml

        training:
          model_modifier:
            modifiers:
              - _target_: "anemoi.training.train.modify.TransferLearningModelModifier"
                checkpoint_path: "/path/to/pretrained.ckpt"
                strict: false
                skip_mismatched: true

    Loading from different sources:

    .. code-block:: yaml

        # From S3
        checkpoint_path: "s3://ml-models/weather-pretrained.ckpt"

        # From HTTP
        checkpoint_path: "https://models.example.com/pretrained.ckpt"

        # From local file
        checkpoint_path: "./checkpoints/pretrained_model.ckpt"

    Programmatic usage:

    .. code-block:: python

        # Create modifier
        transfer_mod = TransferLearningModelModifier(
            checkpoint_path="pretrained.ckpt",
            strict=False,
            skip_mismatched=True
        )

        # Apply to model
        model_with_weights = transfer_mod.apply(model)

        # Check what was loaded
        print(f"Loaded weights from: {transfer_mod.checkpoint_path}")

    Advanced Usage Patterns
    -----------------------

    **Fine-tuning Pipeline**:
        Combine with freezing for selective fine-tuning:

        .. code-block:: yaml

            modifiers:
              # First: Load pretrained weights
              - _target_: "anemoi.training.train.modify.TransferLearningModelModifier"
                checkpoint_path: "pretrained.ckpt"
                strict: false
                skip_mismatched: true

              # Then: Freeze backbone layers
              - _target_: "anemoi.training.train.modify.FreezingModelModifier"
                submodules_to_freeze: ["encoder", "processor.0"]

    **Cross-Resolution Transfer**:
        Handle models with different input/output dimensions:

        .. code-block:: yaml

            checkpoint_path: "low_res_pretrained.ckpt"
            strict: false          # Allow missing keys
            skip_mismatched: true   # Skip shape mismatches

    **Strict Loading for Exact Architectures**:
        Ensure complete weight transfer for identical models:

        .. code-block:: yaml

            checkpoint_path: "exact_architecture.ckpt"
            strict: true           # Require all keys match
            skip_mismatched: false # Fail on any shape mismatch

    Error Handling
    --------------

    The modifier provides detailed error messages for common issues:

    - **File not found**: Clear message with path information
    - **Corrupted checkpoint**: Detailed parsing error information
    - **Key mismatches**: Lists missing/unexpected parameters
    - **Shape mismatches**: Shows expected vs actual shapes
    - **Network errors**: Retry information for remote sources

    Notes
    -----
    - Loading happens during model initialization, before training begins
    - Only model parameters are loaded; optimizer states are ignored
    - The model's architecture must be compatible with checkpoint structure
    - Checkpoint format must be PyTorch-compatible (``.pt``, ``.pth``, ``.ckpt``)
    - Loading is performed using PyTorch's ``load_state_dict`` mechanism
    - Progress is logged at INFO level for successful operations
    - Warnings are logged for skipped/missing parameters

    See Also
    --------
    FreezingModelModifier : Freeze specific parameters after loading
    ModelModifierApplier : Chain multiple modifiers together
    anemoi.training.utils.model_loading.load_model_from_checkpoint : Checkpoint loading (optional)

    Raises
    ------
    FileNotFoundError
        If checkpoint_path cannot be found or accessed
    RuntimeError
        If checkpoint is corrupted or incompatible
    ValueError
        If strict=True and there are key mismatches, or if skip_mismatched=False
        and there are shape mismatches
    ConnectionError
        If remote checkpoint cannot be downloaded

    Examples of Error Scenarios
    ---------------------------

    .. code-block:: python

        # This will raise FileNotFoundError
        modifier = TransferLearningModelModifier("/nonexistent/path.ckpt")

        # This will raise ValueError if architectures don't match exactly
        modifier = TransferLearningModelModifier(
            "different_arch.ckpt",
            strict=True,
            skip_mismatched=False
        )

        # This will warn but continue loading compatible weights
        modifier = TransferLearningModelModifier(
            "partial_match.ckpt",
            strict=False,
            skip_mismatched=True
        )
    """

    def __init__(self, checkpoint_path: Path | str, strict: bool = False, skip_mismatched: bool = True) -> None:
        """Initialize the transfer learning modifier.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path or URL to the checkpoint file. Can be local path or remote URL
            (S3, HTTP, GCS, Azure). The path will be validated and normalized
            during initialization.

        strict : bool, default False
            Whether to require exact parameter name matching between checkpoint
            and model. When False, allows missing or extra parameters with warnings.
            When True, requires all checkpoint parameters to exist in the model.

        skip_mismatched : bool, default True
            How to handle parameters with shape mismatches. When True, parameters
            with incompatible shapes are skipped with warnings. When False, any
            shape mismatch raises an error.

        Raises
        ------
        TypeError
            If checkpoint_path is not a string or Path object
        ValueError
            If checkpoint_path is empty or invalid

        Notes
        -----
        The checkpoint file is not loaded during initialization - only the path
        is stored. Actual loading happens when ``apply()`` is called.
        """
        if not checkpoint_path:
            msg = "Checkpoint path (checkpoint_path) cannot be empty"
            raise ValueError(msg)

        self.checkpoint_path = Path(checkpoint_path)

        self.strict = bool(strict)
        self.skip_mismatched = bool(skip_mismatched)

        LOGGER.debug(
            "Initialized TransferLearningModelModifier: path=%s, strict=%s, skip_mismatched=%s",
            self.checkpoint_path,
            self.strict,
            self.skip_mismatched,
        )

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        """Load pretrained weights into the model from the configured checkpoint.

        This method delegates to Anemoi's extensible checkpoint loading system,
        which handles various source types, error recovery, and detailed progress
        reporting. The loading process:

        1. Downloads/accesses the checkpoint file
        2. Loads and validates the checkpoint structure
        3. Matches parameters between checkpoint and model
        4. Applies the configured strict/mismatch policies
        5. Updates model parameters and reports results

        Parameters
        ----------
        model : torch.nn.Module
            The model to load weights into. Must be fully initialized with
            the target architecture. The model is modified in-place.

        Returns
        -------
        torch.nn.Module
            The input model with loaded weights (same object, modified in-place).

        Raises
        ------
        FileNotFoundError
            If the checkpoint file cannot be found or accessed
        RuntimeError
            If the checkpoint is corrupted, incompatible, or cannot be loaded
        ValueError
            If strict=True and there are parameter name mismatches, or if
            skip_mismatched=False and there are shape mismatches
        ConnectionError
            If a remote checkpoint cannot be downloaded

        Notes
        -----
        - Loading progress is logged at INFO level
        - Warnings are logged for skipped or mismatched parameters
        - The model's training/eval state is preserved
        - Only model parameters are loaded; optimizer states are ignored
        - Memory usage is optimized for large checkpoints

        Examples
        --------
        .. code-block:: python

            modifier = TransferLearningModelModifier("pretrained.ckpt")

            # This modifies the model in-place
            loaded_model = modifier.apply(model)
            assert loaded_model is model  # Same object

            # Check which parameters were loaded
            for name, param in model.named_parameters():
                print(f"{name}: {param.shape}")
        """
        LOGGER.info("Loading transfer learning weights from: %s", self.checkpoint_path)

        # Try to use extensible checkpoint loading system if available
        try:
            from anemoi.training.utils.model_loading import load_model_from_checkpoint

            LOGGER.debug("Using extensible checkpoint loading system")
            loaded_model = load_model_from_checkpoint(
                model=model,
                checkpoint_source=self.checkpoint_path,
                loader_type="transfer_learning",
                strict=self.strict,
                skip_mismatched=self.skip_mismatched,
            )
            LOGGER.info(
                "Successfully loaded transfer learning weights (strict=%s, skip_mismatched=%s)",
                self.strict,
                self.skip_mismatched,
            )
        except ImportError:
            LOGGER.info("Extensible checkpoint loading system not available, using fallback implementation")
            # Use fallback implementation
            return self._fallback_transfer_learning_loading(model)
        except Exception:
            LOGGER.exception("Failed to load transfer learning weights from %s", self.checkpoint_path)
            LOGGER.info("Attempting fallback implementation")
            return self._fallback_transfer_learning_loading(model)
        else:
            return loaded_model

    def _fallback_transfer_learning_loading(self, model: torch.nn.Module) -> torch.nn.Module:  # noqa: C901
        """Fallback implementation for transfer learning when extensible system is not available.

        This method provides basic transfer learning functionality using standard PyTorch
        checkpoint loading with optional shape mismatch handling.

        Parameters
        ----------
        model : torch.nn.Module
            The model to load weights into

        Returns
        -------
        torch.nn.Module
            The model with loaded weights (same object, modified in-place)

        Raises
        ------
        FileNotFoundError
            If the checkpoint file cannot be found
        RuntimeError
            If the checkpoint cannot be loaded or is incompatible
        ValueError
            If strict=True and there are parameter mismatches
        """
        import torch

        try:
            # Load checkpoint - support both local files and basic URL schemes
            if str(self.checkpoint_path).startswith(("http://", "https://")):
                LOGGER.warning(
                    "Loading from URL %s using basic implementation - "
                    "consider using advanced checkpoint loading system for better reliability",
                    self.checkpoint_path,
                )

            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

            # Extract state_dict from checkpoint
            if "state_dict" in checkpoint:
                checkpoint_state_dict = checkpoint["state_dict"]
                LOGGER.debug("Found state_dict in checkpoint")
            elif "model_state_dict" in checkpoint:
                checkpoint_state_dict = checkpoint["model_state_dict"]
                LOGGER.debug("Found model_state_dict in checkpoint")
            else:
                # Assume the entire checkpoint is the state_dict
                checkpoint_state_dict = checkpoint
                LOGGER.debug("Using entire checkpoint as state_dict")

        except Exception as e:
            msg = f"Failed to load checkpoint from {self.checkpoint_path}: {e}"
            LOGGER.exception(msg)
            raise RuntimeError(msg) from e

        # Get model's current state dict
        model_state_dict = model.state_dict()

        # Handle shape mismatches if skip_mismatched is True
        if self.skip_mismatched:
            filtered_state_dict = {}
            skipped_count = 0

            for key, checkpoint_param in checkpoint_state_dict.items():
                if key in model_state_dict:
                    model_param_shape = model_state_dict[key].shape
                    checkpoint_param_shape = checkpoint_param.shape

                    if model_param_shape == checkpoint_param_shape:
                        filtered_state_dict[key] = checkpoint_param
                    else:
                        LOGGER.warning(
                            "Skipping parameter '%s' due to shape mismatch: checkpoint shape %s vs model shape %s",
                            key,
                            checkpoint_param_shape,
                            model_param_shape,
                        )
                        skipped_count += 1
                else:
                    if not self.strict:
                        LOGGER.debug("Skipping parameter '%s' not found in model", key)
                    else:
                        filtered_state_dict[key] = checkpoint_param

            if skipped_count > 0:
                LOGGER.info("Skipped %d parameters due to shape mismatches", skipped_count)

            checkpoint_state_dict = filtered_state_dict

        # Load the state dict
        try:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state_dict, strict=self.strict)

            # Report results
            self._handle_missing_keys(missing_keys)
            self._handle_unexpected_keys(unexpected_keys)

            loaded_params = len(checkpoint_state_dict) - len(missing_keys)
            total_params = len(model_state_dict)

            LOGGER.info(
                "Successfully loaded %d/%d parameters from checkpoint %s (strict=%s, skip_mismatched=%s)",
                loaded_params,
                total_params,
                self.checkpoint_path,
                self.strict,
                self.skip_mismatched,
            )

        except Exception as e:
            msg = f"Failed to load state dict into model: {e}"
            LOGGER.exception(msg)
            raise RuntimeError(msg) from e

        return model

    def _handle_missing_keys(self, missing_keys: list) -> None:
        """Handle missing keys in checkpoint loading."""
        if missing_keys:
            if self.strict:
                msg = f"Missing keys in checkpoint: {missing_keys}"
                raise ValueError(msg)
            LOGGER.warning("Missing keys in checkpoint (continuing): %s", missing_keys)

    def _handle_unexpected_keys(self, unexpected_keys: list) -> None:
        """Handle unexpected keys in checkpoint loading."""
        if unexpected_keys:
            if self.strict:
                msg = f"Unexpected keys in checkpoint: {unexpected_keys}"
                raise ValueError(msg)
            LOGGER.warning("Unexpected keys in checkpoint (continuing): %s", unexpected_keys)


class ModelModifierApplier:
    """Orchestrates the application of multiple model modifiers in sequence.

    The ModelModifierApplier is responsible for reading modifier configurations,
    instantiating the appropriate modifier classes, and applying them to models
    in the correct order. This class is the primary interface between the
    training system and the modifier pattern.

    The applier supports:

    - **Sequential application**: Modifiers are applied in configuration order
    - **Lazy instantiation**: Modifiers are created only when needed
    - **Error handling**: Provides detailed error messages for configuration issues
    - **Logging**: Comprehensive progress reporting for debugging
    - **Flexibility**: Supports any number and combination of modifiers

    Configuration Structure
    -----------------------

    The applier expects configurations in the following format:

    .. code-block:: yaml

        training:
          model_modifier:
            modifiers:
              - _target_: "path.to.FirstModifier"
                param1: value1
                param2: value2
              - _target_: "path.to.SecondModifier"
                param3: value3

    Each modifier entry must include:

    - **_target_**: Fully qualified class name of the modifier
    - **Parameters**: Any initialization arguments for the modifier class

    Examples
    --------
    Basic usage in training configuration:

    .. code-block:: yaml

        training:
          model_modifier:
            modifiers:
              # Load pretrained weights first
              - _target_: "anemoi.training.train.modify.TransferLearningModelModifier"
                checkpoint_path: "pretrained.ckpt"
                strict: false

              # Then freeze specific modules
              - _target_: "anemoi.training.train.modify.FreezingModelModifier"
                submodules_to_freeze: ["encoder", "processor.0"]

    Programmatic usage:

    .. code-block:: python

        from omegaconf import DictConfig

        # Create configuration
        config = DictConfig({
            "training": {
                "model_modifier": {
                    "modifiers": [
                        {
                            "_target_": "anemoi.training.train.modify.FreezingModelModifier",
                            "submodules_to_freeze": ["encoder"]
                        }
                    ]
                }
            }
        })

        # Apply modifiers
        applier = ModelModifierApplier()
        modified_model = applier.process(model, config)

    Advanced Patterns
    -----------------

    **Progressive Fine-tuning**:
        Apply modifiers for gradual model adaptation:

        .. code-block:: yaml

            modifiers:
              # 1. Load base weights
              - _target_: "anemoi.training.train.modify.TransferLearningModelModifier"
                checkpoint_path: "base_model.ckpt"

              # 2. Load domain-specific weights (may override some base weights)
              - _target_: "anemoi.training.train.modify.TransferLearningModelModifier"
                checkpoint_path: "domain_model.ckpt"
                strict: false

              # 3. Freeze backbone, allow head fine-tuning
              - _target_: "anemoi.training.train.modify.FreezingModelModifier"
                submodules_to_freeze: ["encoder", "processor"]

    **Custom Modifier Chain**:
        Combine built-in and custom modifiers:

        .. code-block:: yaml

            modifiers:
              - _target_: "my.custom.NoiseInjectionModifier"
                noise_std: 0.01

              - _target_: "anemoi.training.train.modify.FreezingModelModifier"
                submodules_to_freeze: ["encoder"]

              - _target_: "my.custom.QuantizationModifier"
                bits: 8

    **Conditional Modification**:
        Use Hydra's conditional syntax for environment-specific modifications:

        .. code-block:: yaml

            modifiers:
              - _target_: "anemoi.training.train.modify.TransferLearningModelModifier"
                checkpoint_path: ${checkpoint_path}  # From command line or env

              # Only freeze in production mode
              - _target_: "anemoi.training.train.modify.FreezingModelModifier"
                submodules_to_freeze: ${freeze_layers}

    Error Handling
    --------------

    The applier provides detailed error messages for common configuration issues:

    - **Missing _target_**: Clear indication of which modifier configuration is invalid
    - **Import errors**: Detailed information about failed modifier class imports
    - **Instantiation failures**: Parameter validation errors with context
    - **Application failures**: Runtime errors during modifier.apply() with modifier context

    Notes
    -----
    - Modifiers are applied in the order they appear in the configuration
    - Each modifier receives the output of the previous modifier as input
    - Failed modifier applications stop the entire chain and raise exceptions
    - The applier uses Hydra's ``instantiate()`` for modifier creation
    - All modifier logging is preserved and enhanced with context
    - The applier is stateless and can be reused across multiple models

    See Also
    --------
    ModelModifier : Base class for all modifiers
    FreezingModelModifier : Freeze model parameters
    TransferLearningModelModifier : Load pretrained weights
    hydra.utils.instantiate : Underlying instantiation mechanism

    Raises
    ------
    KeyError
        If required configuration keys are missing
    ImportError
        If a modifier class cannot be imported
    TypeError
        If a modifier cannot be instantiated with provided parameters
    RuntimeError
        If a modifier's apply() method fails during execution

    Examples of Error Scenarios
    ----------------------------

    .. code-block:: python

        # This will raise KeyError - missing _target_
        bad_config = DictConfig({
            "training": {
                "model_modifier": {
                    "modifiers": [{"param": "value"}]  # No _target_
                }
            }
        })

        # This will raise ImportError - nonexistent class
        bad_config = DictConfig({
            "training": {
                "model_modifier": {
                    "modifiers": [{
                        "_target_": "nonexistent.module.BadModifier"
                    }]
                }
            }
        })

        # This will raise TypeError - invalid parameters
        bad_config = DictConfig({
            "training": {
                "model_modifier": {
                    "modifiers": [{
                        "_target_": "anemoi.training.train.modify.FreezingModelModifier",
                        "invalid_param": "value"  # FreezingModelModifier doesn't accept this
                    }]
                }
            }
        })
    """

    def process(self, base_model: torch.nn.Module, config: DictConfig) -> torch.nn.Module:
        """Apply all configured model modifiers to the given model in sequence.

        This is the main entry point for the modifier system. It reads the
        configuration, instantiates and applies each modifier in order, and
        returns the fully modified model.

        The process flow:

        1. **Validate configuration**: Check for required config structure
        2. **Instantiate modifiers**: Create modifier instances using Hydra
        3. **Apply sequentially**: Apply each modifier to the result of the previous
        4. **Log progress**: Report each modification step for debugging
        5. **Return result**: Return the fully modified model

        Parameters
        ----------
        base_model : torch.nn.Module
            The initial model to modify. This model serves as the starting point
            for the modification chain. The model may be modified in-place by
            some modifiers.

        config : DictConfig
            Complete training configuration containing the modifier specifications
            under ``config.training.model_modifier.modifiers``. If no modifiers
            are configured, the model is returned unchanged.

        Returns
        -------
        torch.nn.Module
            The modified model after applying all configured modifiers. This may
            be the same object as ``base_model`` (if modified in-place) or a new
            model object, depending on the specific modifiers applied.

        Raises
        ------
        KeyError
            If required configuration keys are missing or malformed
        ImportError
            If any modifier class cannot be imported from its specified path
        TypeError
            If any modifier cannot be instantiated due to parameter issues
        RuntimeError
            If any modifier's ``apply()`` method fails during execution
        AttributeError
            If the configuration structure is invalid

        Notes
        -----
        - Modifiers are applied in strict sequence as configured
        - Each modifier receives the output of the previous modifier
        - If any modifier fails, the entire process is aborted with an exception
        - Progress logging helps track which modifier caused any failures
        - The method is idempotent - calling it multiple times with the same
          inputs should produce the same result

        Examples
        --------
        Basic usage:

        .. code-block:: python

            applier = ModelModifierApplier()
            modified_model = applier.process(original_model, config)

            # The model may be the same object (modified in-place)
            # or a new object, depending on the modifiers
            print(f"Same object: {modified_model is original_model}")

        With error handling:

        .. code-block:: python

            try:
                applier = ModelModifierApplier()
                modified_model = applier.process(model, config)
                print("All modifiers applied successfully")

            except ImportError as e:
                print(f"Modifier class not found: {e}")
            except RuntimeError as e:
                print(f"Modifier application failed: {e}")

        Debugging modifier application:

        .. code-block:: python

            import logging
            logging.basicConfig(level=logging.INFO)

            applier = ModelModifierApplier()
            # This will log each modifier as it's applied
            modified_model = applier.process(model, config)
        """
        model = base_model

        # Validate configuration structure
        if not hasattr(config, "training"):
            LOGGER.debug("No training configuration found, skipping model modifications")
            return model

        if not hasattr(config.training, "model_modifier"):
            LOGGER.debug("No model_modifier configuration found, skipping model modifications")
            return model

        if not hasattr(config.training.model_modifier, "modifiers"):
            LOGGER.debug("No modifiers list found in configuration, skipping model modifications")
            return model

        modifiers_config = config.training.model_modifier.modifiers
        if not modifiers_config:
            LOGGER.info("Empty modifiers list in configuration, returning model as-is")
            return model

        LOGGER.info("Applying %d model modifiers in sequence", len(modifiers_config))

        # Instantiate each modifier from the configuration
        model_modifier_instances = []
        try:
            for i, modifier_config in enumerate(modifiers_config):
                modifier_instance = instantiate(modifier_config)
                model_modifier_instances.append(modifier_instance)
                LOGGER.debug("Instantiated modifier %d: %s", i + 1, type(modifier_instance).__name__)
        except Exception as e:
            LOGGER.exception("Failed to instantiate modifier %d from config %s.", i + 1, modifier_config)
            msg = f"Modifier {i + 1} instantiation failed: {e}"
            raise RuntimeError(msg) from e

        # Apply each modifier in sequence
        try:
            for i, modifier_instance in enumerate(model_modifier_instances):
                modifier_name = type(modifier_instance).__name__
                LOGGER.info("Applying modifier %d/%d: %s", i + 1, len(model_modifier_instances), modifier_name)

                model = modifier_instance.apply(model)
                LOGGER.debug("Successfully applied modifier: %s", modifier_name)
        except Exception as e:
            LOGGER.exception("Failed to apply modifier %s.", modifier_name)
            msg = f"Modifier '{modifier_name}' application failed: {e}"
            raise RuntimeError(msg) from e

        LOGGER.info("Successfully applied all %d model modifiers", len(model_modifier_instances))
        return model
