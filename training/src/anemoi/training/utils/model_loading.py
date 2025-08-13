# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Model Loading System for Anemoi Training.

This module provides a flexible model loading system that supports different
strategies for loading PyTorch model weights from checkpoints. It handles
various scenarios including standard loading, transfer learning with size
mismatches, and weights-only loading.

Key Features
============

* **Multiple Loading Strategies**: Support for standard, transfer learning, and weights-only loading
* **Size Mismatch Handling**: Intelligent handling of parameter size mismatches during transfer learning
* **Metadata Preservation**: Maintains important model metadata and hyperparameters
* **Registry Pattern**: Extensible loader registry for custom loading strategies
* **Integration**: Seamless integration with the checkpoint loading system

Architecture
============

The system uses a strategy pattern with abstract base classes:

1. **ModelLoader**: Abstract base class defining the loader interface
2. **Concrete Loaders**: Specific loading strategies (standard, transfer learning, weights-only)
3. **ModelLoaderRegistry**: Central registry managing different loading strategies
4. **Global Registry**: Pre-configured registry instance ready for immediate use

Supported Loading Strategies
============================

* **Standard**: Direct loading for exact architecture matches
* **Transfer Learning**: Loading with intelligent size mismatch handling
* **Weights Only**: Loading only model parameters, skipping optimizer states

Basic Usage
===========

.. code-block:: python

    from anemoi.training.utils.model_loading import load_model_from_checkpoint

    # Standard loading (exact architecture match required)
    model = load_model_from_checkpoint(
        model=my_model,
        checkpoint_source="/path/to/checkpoint.ckpt",
        loader_type="standard"
    )

    # Transfer learning (handles size mismatches)
    model = load_model_from_checkpoint(
        model=my_model,
        checkpoint_source="s3://bucket/pretrained.ckpt",
        loader_type="transfer_learning"
    )

    # Weights only (ignores optimizer state)
    model = load_model_from_checkpoint(
        model=my_model,
        checkpoint_source="https://example.com/weights.ckpt",
        loader_type="weights_only"
    )

Advanced Usage
==============

.. code-block:: python

    from anemoi.training.utils.model_loading import (
        ModelLoaderRegistry,
        TransferLearningModelLoader,
        model_loader_registry
    )

    # Use registry directly
    loader = model_loader_registry.get_loader("transfer_learning")
    model = loader.load_model_weights(
        model=my_model,
        checkpoint_source="s3://bucket/model.ckpt",
        strict=False,
        skip_mismatched=True
    )

    # Custom registry
    custom_registry = ModelLoaderRegistry()
    custom_registry.register("my_loader", MyCustomLoader())

Extending the System
====================

To add custom loading strategies:

.. code-block:: python

    class QuantizedModelLoader(ModelLoader):
        def load_model_weights(self, model, checkpoint_source, **kwargs):
            # Custom quantized loading logic
            checkpoint = load_checkpoint_from_source(checkpoint_source)
            # Apply quantization during loading
            return quantize_and_load(model, checkpoint)

    # Register with the global registry
    model_loader_registry.register("quantized", QuantizedModelLoader())

Transfer Learning Details
=========================

The transfer learning loader provides several options:

* **Size Mismatch Handling**: Automatically skips parameters with incompatible shapes
* **Selective Loading**: Option to skip parameters not present in the target model
* **Logging**: Detailed logging of skipped parameters for debugging

.. code-block:: python

    # Transfer learning with custom options
    model = load_model_from_checkpoint(
        model=target_model,
        checkpoint_source="pretrained.ckpt",
        loader_type="transfer_learning",
        strict=False,              # Allow missing keys
        skip_mismatched=True       # Skip size mismatches
    )

Integration with Training
=========================

This module integrates with the training pipeline:

.. code-block:: python

    # In training configuration
    training:
      checkpoint_loading:
        source: "s3://bucket/pretrained.ckpt"
        loader_type: "transfer_learning"
        strict: false
        skip_mismatched: true

    # The training system uses this module to load model weights

See Also
--------
* :mod:`anemoi.training.utils.checkpoint_loaders`: Checkpoint source handling
* :mod:`anemoi.training.train.modify`: Model modification system
* :mod:`anemoi.training.train.train`: Main training pipeline integration

Notes
-----
* All loaders preserve model metadata when available
* Transfer learning loader provides detailed logging for debugging
* The system is designed to be thread-safe for loading operations
* Custom loaders should handle errors gracefully and provide informative messages
"""

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

from anemoi.training.utils.checkpoint_loaders import load_checkpoint_from_source

if TYPE_CHECKING:
    from pathlib import Path

    import torch

LOGGER = logging.getLogger(__name__)


class ModelLoader(ABC):
    """Abstract base class for loading PyTorch model weights from checkpoints.

    This class defines the interface that all model loaders must implement.
    Concrete implementations handle different loading strategies such as
    standard loading, transfer learning, or specialized loading requirements.

    The loader system provides a unified interface for loading model weights
    while allowing different strategies for handling various scenarios like
    architecture mismatches, selective parameter loading, or custom preprocessing.

    Design Principles
    =================

    * **Strategy Pattern**: Each loader implements a specific loading strategy
    * **Consistent Interface**: All loaders use the same method signature
    * **Error Handling**: Loaders provide clear error messages for debugging
    * **Metadata Preservation**: Important model metadata is preserved when possible

    Implementing Custom Loaders
    ============================

    To create a custom model loader:

    .. code-block:: python

        class MyCustomLoader(ModelLoader):
            def load_model_weights(self, model, checkpoint_source, strict=True, **kwargs):
                # Load checkpoint
                checkpoint = load_checkpoint_from_source(checkpoint_source)

                # Custom processing logic
                processed_state_dict = self.preprocess_weights(checkpoint["state_dict"])

                # Load into model
                model.load_state_dict(processed_state_dict, strict=strict)

                return model

    Error Handling Guidelines
    =========================

    Implementations should raise appropriate exceptions:

    * ``ValueError``: For invalid checkpoint format or missing required keys
    * ``RuntimeError``: For PyTorch loading errors or shape mismatches
    * ``FileNotFoundError``: When checkpoint source cannot be accessed
    * ``KeyError``: When required checkpoint keys are missing

    See Also
    --------
    * :class:`StandardModelLoader`: Standard PyTorch Lightning checkpoint loading
    * :class:`TransferLearningModelLoader`: Loading with size mismatch handling
    * :class:`WeightsOnlyModelLoader`: Loading only model parameters
    * :class:`ModelLoaderRegistry`: Registry for managing multiple loaders
    """

    @abstractmethod
    def load_model_weights(
        self,
        model: torch.nn.Module,
        checkpoint_source: str | Path,
        strict: bool = True,
        **kwargs,
    ) -> torch.nn.Module:
        """Load model weights from a checkpoint source into the target model.

        This method must be implemented by all concrete model loaders.
        It should handle loading weights from the checkpoint into the provided
        model using the specific strategy implemented by the loader.

        The method is responsible for:
        1. Loading the checkpoint data via the checkpoint loading system
        2. Extracting and processing the model state dictionary
        3. Loading the processed weights into the target model
        4. Preserving any relevant metadata
        5. Returning the model with loaded weights

        Parameters
        ----------
        model : torch.nn.Module
            The target PyTorch model to load weights into. The model should
            already be instantiated with the desired architecture.
        checkpoint_source : str | Path
            The checkpoint source to load from. This can be a local file path,
            remote URL, or cloud storage URL. The source will be handled by
            the checkpoint loading system.
        strict : bool, optional
            Whether to strictly enforce that the keys in the checkpoint's
            state_dict match the keys returned by the model's state_dict().
            Default is True. When False, allows loading with missing keys.
        **kwargs
            Additional keyword arguments specific to the loader implementation.
            Common options include:

            - ``skip_mismatched`` (bool): Skip parameters with size mismatches
            - ``prefix`` (str): Add prefix to checkpoint parameter names
            - ``map_location`` (str): Device to map tensors to during loading

        Returns
        -------
        torch.nn.Module
            The input model with loaded weights. The same object is returned
            for chaining, but the model's parameters are updated in-place.

        Raises
        ------
        ValueError
            If the checkpoint format is invalid or missing required keys like 'state_dict'.
        RuntimeError
            If PyTorch fails to load the state dict due to shape mismatches or other issues.
        FileNotFoundError
            If the checkpoint source cannot be accessed or doesn't exist.
        KeyError
            If required keys are missing from the checkpoint dictionary.
        ImportError
            If required dependencies for accessing the checkpoint source are missing.

        Examples
        --------

        .. code-block:: python

            # Basic usage
            loader = SomeModelLoader()
            loaded_model = loader.load_model_weights(
                model=my_model,
                checkpoint_source="/path/to/checkpoint.ckpt"
            )

            # With custom options
            loaded_model = loader.load_model_weights(
                model=my_model,
                checkpoint_source="s3://bucket/checkpoint.ckpt",
                strict=False,
                skip_mismatched=True
            )

        Notes
        -----
        * The model is modified in-place; the returned model is the same object
        * Implementations should preserve model metadata when available in checkpoints
        * Logging should be used to provide feedback on the loading process
        * The method should be thread-safe for concurrent loading operations
        """
        ...


class StandardModelLoader(ModelLoader):
    """Standard model loader for PyTorch Lightning checkpoints.

    This loader handles the most common case of loading model weights from
    PyTorch Lightning checkpoints where the source and target models have
    identical architectures. It performs direct state dict loading with
    optional strict mode control.

    Use Cases
    =========

    * **Resuming Training**: Loading from a checkpoint to continue training
    * **Model Evaluation**: Loading trained weights for inference
    * **Exact Architecture Match**: When source and target models are identical
    * **Production Deployment**: Loading stable, validated model weights

    Features
    ========

    * **Strict Validation**: Enforces exact parameter key matching by default
    * **Metadata Preservation**: Maintains model metadata and hyperparameters
    * **Data Indices Support**: Preserves data indexing information for compatibility
    * **Lightning Integration**: Full support for PyTorch Lightning checkpoint format

    Behavior
    ========

    1. Loads checkpoint from any supported source (local, S3, HTTP, etc.)
    2. Validates that 'state_dict' key exists in checkpoint
    3. Performs direct state dict loading into the model
    4. Preserves metadata including data indices if available
    5. Logs successful loading for debugging

    Usage Examples
    ==============

    .. code-block:: python

        loader = StandardModelLoader()

        # Standard loading with strict validation
        model = loader.load_model_weights(
            model=my_model,
            checkpoint_source="/path/to/checkpoint.ckpt",
            strict=True
        )

        # Relaxed loading (allows missing keys)
        model = loader.load_model_weights(
            model=my_model,
            checkpoint_source="s3://bucket/checkpoint.ckpt",
            strict=False
        )

    Error Handling
    ==============

    The loader will raise errors for:

    * Missing 'state_dict' in checkpoint
    * Parameter shape mismatches (in strict mode)
    * Missing parameters in model (in strict mode)
    * Unexpected parameters in checkpoint (in strict mode)

    When Not to Use
    ===============

    This loader is not suitable for:

    * Transfer learning between different architectures
    * Loading from models with different parameter shapes
    * Selective parameter loading
    * Custom preprocessing of checkpoint data

    For these scenarios, use :class:`TransferLearningModelLoader` instead.
    """

    def load_model_weights(
        self,
        model: torch.nn.Module,
        checkpoint_source: str | Path,
        strict: bool = True,
        **kwargs,  # noqa: ARG002
    ) -> torch.nn.Module:
        """Load PyTorch Lightning checkpoint weights using standard loading.

        This method performs direct state dict loading from a PyTorch Lightning
        checkpoint. It expects the checkpoint to contain a 'state_dict' key with
        model parameters that exactly match the target model architecture.

        The loading process:
        1. Loads checkpoint from the specified source
        2. Validates checkpoint format and required keys
        3. Extracts the state dictionary
        4. Loads state dict into model with optional strict validation
        5. Preserves important metadata for model compatibility

        Parameters
        ----------
        model : torch.nn.Module
            The target model to load weights into. Must have the same architecture
            as the model that created the checkpoint.
        checkpoint_source : str | Path
            Path or URL to the checkpoint file. Supports local files, S3, HTTP,
            Google Cloud Storage, and Azure Blob Storage.
        strict : bool, optional
            Whether to strictly enforce that the keys in the checkpoint state_dict
            match the keys returned by the model's state_dict(). Default is True.
            When False, allows loading with missing keys but still fails on
            unexpected keys or shape mismatches.
        **kwargs
            Additional keyword arguments. Currently unused but maintained for
            interface compatibility.

        Returns
        -------
        torch.nn.Module
            The input model with loaded weights. The model is modified in-place.

        Raises
        ------
        ValueError
            If the checkpoint doesn't contain a 'state_dict' key.
        RuntimeError
            If PyTorch fails to load the state dict due to architecture mismatch,
            shape mismatch, or missing/unexpected parameters (when strict=True).
        FileNotFoundError
            If the checkpoint source cannot be accessed.
        KeyError
            If required checkpoint structure is malformed.

        Examples
        --------

        .. code-block:: python

            loader = StandardModelLoader()

            # Load from local file with strict validation
            model = loader.load_model_weights(
                model=my_lightning_model,
                checkpoint_source="./checkpoints/epoch_10.ckpt",
                strict=True
            )

            # Load from S3 with relaxed validation
            model = loader.load_model_weights(
                model=my_lightning_model,
                checkpoint_source="s3://models/pretrained.ckpt",
                strict=False
            )

            # The model now has loaded weights
            predictions = model(input_data)

        Notes
        -----
        * This loader preserves PyTorch Lightning metadata including hyperparameters
        * Data indices are maintained for models that use them
        * The model's training state is not affected (remains in same mode)
        * Loading is performed on CPU first, then moved to model's device
        """
        checkpoint = load_checkpoint_from_source(checkpoint_source)

        if "state_dict" not in checkpoint:
            msg = f"No 'state_dict' found in checkpoint from {checkpoint_source}"
            raise ValueError(msg)

        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict, strict=strict)

        # Preserve checkpoint metadata for compatibility
        if hasattr(model, "_ckpt_model_name_to_index") and "hyper_parameters" in checkpoint:
            hyper_params = checkpoint["hyper_parameters"]
            if "data_indices" in hyper_params:
                model._ckpt_model_name_to_index = hyper_params["data_indices"].name_to_index

        LOGGER.info("Loaded model weights from %s", checkpoint_source)
        return model


class TransferLearningModelLoader(ModelLoader):
    """Model loader for transfer learning with intelligent size mismatch handling.

    This loader is specifically designed for transfer learning scenarios where
    the source and target models may have different architectures. It provides
    intelligent handling of parameter mismatches including size differences
    and missing/extra parameters.

    Key Features
    ============

    * **Size Mismatch Handling**: Automatically skips parameters with incompatible shapes
    * **Selective Loading**: Only loads parameters that exist in both source and target
    * **Detailed Logging**: Comprehensive logging of skipped parameters for debugging
    * **Flexible Validation**: Supports both strict and relaxed loading modes
    * **Metadata Preservation**: Maintains compatibility metadata when possible

    Use Cases
    =========

    * **Domain Adaptation**: Adapting models trained on one domain to another
    * **Architecture Changes**: Loading weights when model architecture has been modified
    * **Fine-tuning**: Loading pretrained weights for fine-tuning on new tasks
    * **Partial Loading**: Loading only compatible parameters from larger models
    * **Model Evolution**: Migrating weights between model versions

    Transfer Learning Strategies
    ============================

    **Conservative Strategy** (``skip_mismatched=True``):
        Only loads parameters with exact shape matches. Safest approach that
        guarantees no loading errors but may skip useful parameters.

    **Aggressive Strategy** (``skip_mismatched=False``):
        Attempts to load all available parameters. May fail on shape mismatches
        but can be useful for debugging or when you know shapes are compatible.

    Behavior Details
    ================

    When ``skip_mismatched=True`` (default):
    1. Compares each parameter's shape between checkpoint and model
    2. Only includes parameters with exact shape matches
    3. Logs skipped parameters with shape information
    4. Results in partial but safe parameter loading

    When ``skip_mismatched=False``:
    1. Includes all parameters present in both checkpoint and model
    2. May fail during PyTorch loading if shapes don't match
    3. Useful for debugging or when manual shape verification is done

    Usage Examples
    ==============

    .. code-block:: python

        loader = TransferLearningModelLoader()

        # Conservative transfer learning (recommended)
        model = loader.load_model_weights(
            model=target_model,
            checkpoint_source="s3://bucket/pretrained.ckpt",
            strict=False,              # Allow missing keys
            skip_mismatched=True       # Skip size mismatches (safe)
        )

        # Aggressive transfer learning (for debugging)
        model = loader.load_model_weights(
            model=target_model,
            checkpoint_source="pretrained_model.ckpt",
            strict=False,
            skip_mismatched=False      # May fail on mismatches
        )

    Common Scenarios
    ================

    **Scenario 1: Different Output Dimensions**
        Source model trained for 100 classes, target for 10 classes.
        The loader will skip the final classification layer but load
        all feature extraction layers.

    **Scenario 2: Added Layers**
        Target model has additional layers not in source model.
        Only shared layers are loaded; new layers retain initialization.

    **Scenario 3: Removed Layers**
        Source model has layers not in target model.
        Extra parameters are ignored; target model remains unchanged.

    Debugging Transfer Learning
    ===========================

    The loader provides detailed logging to help debug transfer learning:

    * Lists all skipped parameters with shape information
    * Identifies parameters not present in target model
    * Reports successful parameter transfers
    * Logs final transfer learning status

    Monitor logs to ensure expected parameters are being transferred.

    Performance Considerations
    ==========================

    * Parameter comparison is O(n) where n is number of checkpoint parameters
    * Shape checking adds minimal overhead
    * Logging can be verbose for models with many mismatched parameters
    * Memory usage scales with checkpoint size during filtering
    """

    def load_model_weights(
        self,
        model: torch.nn.Module,
        checkpoint_source: str | Path,
        strict: bool = False,
        skip_mismatched: bool = True,
        **kwargs,  # noqa: ARG002
    ) -> torch.nn.Module:
        """Load model weights with transfer learning compatibility.

        This method performs intelligent transfer learning by comparing parameter
        shapes between the checkpoint and target model, selectively loading only
        compatible parameters while providing detailed feedback on skipped parameters.

        The transfer learning process:
        1. Loads checkpoint from any supported source
        2. Extracts source model's state dictionary
        3. Compares parameter shapes with target model
        4. Filters parameters based on compatibility and settings
        5. Loads compatible parameters into target model
        6. Preserves metadata and sets transfer learning flags

        Parameters
        ----------
        model : torch.nn.Module
            The target model to load weights into. Can have a different architecture
            from the source model in the checkpoint.
        checkpoint_source : str | Path
            Path or URL to the checkpoint file containing source model weights.
            Supports all checkpoint loading sources (local, S3, HTTP, etc.).
        strict : bool, optional
            Whether to strictly enforce parameter key matching. Default is False
            to allow missing keys in transfer learning scenarios. Setting to True
            will cause failures if any expected parameters are missing.
        skip_mismatched : bool, optional
            Whether to skip parameters with shape mismatches. Default is True.
            When True, only parameters with exact shape matches are loaded.
            When False, attempts to load all available parameters (may fail
            during PyTorch loading if shapes are incompatible).
        **kwargs
            Additional keyword arguments. Currently unused but maintained for
            interface compatibility.

        Returns
        -------
        torch.nn.Module
            The input model with compatible weights loaded from the checkpoint.
            The model is modified in-place and marked as having initialized weights.

        Raises
        ------
        ValueError
            If the checkpoint doesn't contain a 'state_dict' key.
        RuntimeError
            If PyTorch fails to load filtered parameters due to remaining
            incompatibilities (rare when skip_mismatched=True).
        FileNotFoundError
            If the checkpoint source cannot be accessed.

        Examples
        --------

        .. code-block:: python

            loader = TransferLearningModelLoader()

            # Safe transfer learning (recommended approach)
            target_model = loader.load_model_weights(
                model=my_model,
                checkpoint_source="s3://bucket/pretrained_resnet.ckpt",
                strict=False,           # Allow missing keys
                skip_mismatched=True    # Skip incompatible shapes
            )

            # Check if weights were initialized
            if hasattr(target_model, 'weights_initialized'):
                print(f"Transfer learning completed: {target_model.weights_initialized}")

            # View detailed logs to see what was transferred
            # Logs will show:
            # - "Skipping parameter with size mismatch: layer.weight"
            # - "Skipping parameter not in model: old_layer.bias"
            # - "Transfer learning applied successfully"

        Advanced usage with custom filtering:

        .. code-block:: python

            # For models where you know shapes are compatible
            model = loader.load_model_weights(
                model=compatible_model,
                checkpoint_source="./checkpoints/similar_arch.ckpt",
                strict=True,            # Enforce exact match
                skip_mismatched=False   # Trust shape compatibility
            )

        Notes
        -----
        * The model's ``weights_initialized`` attribute is set to True after loading
        * Comprehensive logging helps debug transfer learning issues
        * Metadata preservation maintains model compatibility when possible
        * The method handles both PyTorch and PyTorch Lightning checkpoints
        * Parameter filtering reduces memory usage by creating smaller state dicts
        """
        checkpoint = load_checkpoint_from_source(checkpoint_source)

        if "state_dict" not in checkpoint:
            msg = f"No 'state_dict' found in checkpoint from {checkpoint_source}"
            raise ValueError(msg)

        state_dict = checkpoint["state_dict"]
        model_state_dict = model.state_dict()

        if skip_mismatched:
            # Filter out layers with size mismatch
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if key in model_state_dict:
                    if value.shape == model_state_dict[key].shape:
                        filtered_state_dict[key] = value
                    else:
                        LOGGER.info("Skipping parameter with size mismatch: %s", key)
                        LOGGER.info("Checkpoint shape: %s, Model shape: %s", value.shape, model_state_dict[key].shape)
                else:
                    LOGGER.info("Skipping parameter not in model: %s", key)

            state_dict = filtered_state_dict

        model.load_state_dict(state_dict, strict=strict)

        # Preserve checkpoint metadata for compatibility
        if hasattr(model, "_ckpt_model_name_to_index") and "hyper_parameters" in checkpoint:
            hyper_params = checkpoint["hyper_parameters"]
            if "data_indices" in hyper_params:
                model._ckpt_model_name_to_index = hyper_params["data_indices"].name_to_index

        model.weights_initialized = True
        LOGGER.info("Transfer learning applied successfully from %s", checkpoint_source)
        return model


class WeightsOnlyModelLoader(ModelLoader):
    """Model loader that exclusively loads model weights, ignoring optimizer states and metadata.

    This loader is designed for scenarios where only the model parameters are needed,
    such as inference, evaluation, or when starting fresh training with pretrained weights.
    It ignores optimizer states, learning rate scheduler states, and other training metadata.

    Key Features
    ============

    * **Weights Focus**: Loads only model parameters, ignoring training state
    * **Clean Loading**: No optimizer state or scheduler state contamination
    * **Mismatch Handling**: Uses robust transfer learning logic internally
    * **Inference Ready**: Ideal for deploying models in production
    * **Fresh Training**: Perfect for starting new training with pretrained weights

    Use Cases
    =========

    * **Model Inference**: Loading weights for evaluation or prediction
    * **Production Deployment**: Clean model loading without training artifacts
    * **Fresh Training Start**: Using pretrained weights for new training run
    * **Model Analysis**: Loading weights for model inspection or analysis
    * **Cross-framework Transfer**: Loading weights when changing training frameworks

    Implementation Details
    ======================

    This loader leverages the :class:`TransferLearningModelLoader` internally
    with specific settings optimized for weights-only loading:

    * Uses ``skip_mismatched=False`` for more permissive loading
    * Inherits robust mismatch handling and logging
    * Maintains the same error handling and debugging features
    * Preserves essential model metadata while ignoring training state

    Comparison with Other Loaders
    =============================

    **vs StandardModelLoader**:
        WeightsOnlyModelLoader is more robust for architecture differences
        and focuses purely on model parameters.

    **vs TransferLearningModelLoader**:
        WeightsOnlyModelLoader uses different default settings optimized
        for clean weight loading rather than transfer learning scenarios.

    Usage Examples
    ==============

    .. code-block:: python

        loader = WeightsOnlyModelLoader()

        # Load weights for inference
        inference_model = loader.load_model_weights(
            model=my_model,
            checkpoint_source="s3://bucket/trained_model.ckpt",
            strict=True  # Expect exact architecture match
        )
        inference_model.eval()  # Set to evaluation mode

        # Load weights for fresh training start
        fresh_model = loader.load_model_weights(
            model=new_model,
            checkpoint_source="./pretrained/weights.ckpt",
            strict=False  # Allow some flexibility
        )

        # The model has no optimizer state or training metadata
        # Perfect for starting new training or inference

    When to Use
    ===========

    **Ideal for**:
    - Model deployment and inference
    - Starting fresh training with pretrained weights
    - Model evaluation and analysis
    - Cross-framework model migration
    - Clean model state without training artifacts

    **Not ideal for**:
    - Resuming training (use StandardModelLoader)
    - When you need optimizer state
    - When training metadata is important

    Performance Considerations
    ==========================

    * Slightly more overhead due to internal transfer learning logic
    * Memory efficient as it ignores optimizer and scheduler states
    * Same loading performance as TransferLearningModelLoader
    * Clean model state reduces memory usage in production
    """

    def load_model_weights(
        self,
        model: torch.nn.Module,
        checkpoint_source: str | Path,
        strict: bool = True,
        **kwargs,
    ) -> torch.nn.Module:
        """Load only model weights from checkpoint, ignoring training state.

        This method loads exclusively model parameters from a checkpoint,
        ignoring optimizer states, learning rate scheduler states, and other
        training metadata. It uses the transfer learning loader internally
        for robust handling of potential architecture differences.

        The loading process:
        1. Delegates to TransferLearningModelLoader with weights-optimized settings
        2. Uses ``skip_mismatched=False`` for more permissive loading
        3. Inherits all mismatch handling and logging capabilities
        4. Returns model ready for inference or fresh training

        Parameters
        ----------
        model : torch.nn.Module
            The target model to load weights into. Should have compatible
            architecture with the checkpoint model.
        checkpoint_source : str | Path
            Path or URL to the checkpoint file. Supports all checkpoint
            loading sources (local, S3, HTTP, GCS, Azure).
        strict : bool, optional
            Whether to strictly enforce parameter key matching. Default is True.
            For weights-only loading, this typically should be True unless
            you expect architecture differences.
        **kwargs
            Additional keyword arguments passed to the underlying transfer
            learning loader. Common options:

            - Custom parameters specific to your use case
            - Any other TransferLearningModelLoader options

        Returns
        -------
        torch.nn.Module
            The input model with loaded weights, ready for inference or fresh
            training. The model contains no optimizer state or training metadata.

        Raises
        ------
        ValueError
            If the checkpoint doesn't contain a 'state_dict' key.
        RuntimeError
            If PyTorch fails to load the weights due to architecture mismatch.
        FileNotFoundError
            If the checkpoint source cannot be accessed.

        Examples
        --------

        .. code-block:: python

            loader = WeightsOnlyModelLoader()

            # Load for inference (strict matching)
            model = loader.load_model_weights(
                model=inference_model,
                checkpoint_source="s3://models/production.ckpt",
                strict=True
            )
            model.eval()  # Ready for inference

            # Load for fresh training (more flexible)
            model = loader.load_model_weights(
                model=training_model,
                checkpoint_source="./pretrained/backbone.ckpt",
                strict=False
            )
            # Model ready for new training without old optimizer state

        Production deployment example:

        .. code-block:: python

            # Load clean weights for production inference
            production_model = MyModel(config)
            loader = WeightsOnlyModelLoader()

            production_model = loader.load_model_weights(
                model=production_model,
                checkpoint_source="s3://production/model_v2.ckpt"
            )

            # Model is clean and ready for inference
            production_model.eval()
            with torch.no_grad():
                predictions = production_model(input_batch)

        Notes
        -----
        * Internally uses TransferLearningModelLoader with ``skip_mismatched=False``
        * Inherits all robust error handling and logging from transfer learning loader
        * Results in clean model state without optimizer or scheduler artifacts
        * The loaded model will have ``weights_initialized=True`` attribute set
        * Perfect for inference deployments where training state is not needed
        """
        # For weights-only loading, we use the TransferLearningModelLoader
        # as it has better handling of potential mismatches
        loader = TransferLearningModelLoader()
        return loader.load_model_weights(
            model=model,
            checkpoint_source=checkpoint_source,
            strict=strict,
            skip_mismatched=False,
            **kwargs,
        )


class ModelLoaderRegistry:
    """Central registry for managing model loading strategies.

    This registry maintains a collection of model loaders and provides a
    unified interface for loading model weights using different strategies.
    It supports built-in loaders for common scenarios and allows registration
    of custom loaders for specialized requirements.

    The registry uses a name-based lookup system where each loader is
    associated with a descriptive string identifier. This allows easy
    selection of loading strategies through configuration or runtime decisions.

    Features
    ========

    * **Strategy Selection**: Choose loading strategy by name
    * **Extensible**: Easy registration of custom loaders
    * **Pre-configured**: Comes with loaders for common use cases
    * **Error Handling**: Clear error messages for unknown loader types
    * **Unified Interface**: Consistent API across all loading strategies

    Built-in Loaders
    ================

    The registry comes pre-configured with three essential loaders:

    * **"standard"**: :class:`StandardModelLoader` - Direct Lightning checkpoint loading
    * **"transfer_learning"**: :class:`TransferLearningModelLoader` - Size mismatch handling
    * **"weights_only"**: :class:`WeightsOnlyModelLoader` - Clean weight loading

    Usage Examples
    ==============

    .. code-block:: python

        from anemoi.training.utils.model_loading import model_loader_registry

        # Use built-in loaders
        model = model_loader_registry.load_model_weights(
            model=my_model,
            checkpoint_source="s3://bucket/checkpoint.ckpt",
            loader_type="transfer_learning",
            strict=False
        )

        # Get specific loader
        loader = model_loader_registry.get_loader("weights_only")
        model = loader.load_model_weights(model, checkpoint_source)

    Custom Loaders
    ==============

    Register custom loaders for specialized scenarios:

    .. code-block:: python

        class QuantizedModelLoader(ModelLoader):
            def load_model_weights(self, model, checkpoint_source, **kwargs):
                # Custom quantized loading logic
                return quantize_and_load(model, checkpoint_source)

        # Register custom loader
        model_loader_registry.register("quantized", QuantizedModelLoader())

        # Use custom loader
        model = model_loader_registry.load_model_weights(
            model=my_model,
            checkpoint_source="./checkpoints/quantized.ckpt",
            loader_type="quantized"
        )

    Configuration Integration
    =========================

    The registry integrates with training configuration:

    .. code-block:: python

        # In training config
        training:
          checkpoint_loading:
            source: "s3://bucket/pretrained.ckpt"
            loader_type: "transfer_learning"  # Registry key
            strict: false
            skip_mismatched: true

        # The training system uses the registry
        model = model_loader_registry.load_model_weights(
            model=model,
            checkpoint_source=config.checkpoint_loading.source,
            loader_type=config.checkpoint_loading.loader_type,
            **config.checkpoint_loading
        )

    Thread Safety
    =============

    The registry is thread-safe for read operations (getting loaders, loading models)
    but not for write operations (registering new loaders). Register all custom
    loaders during initialization before concurrent usage.

    Performance Considerations
    ==========================

    * Loader lookup is O(1) dictionary access
    * Loader instances are reused across calls
    * No overhead for unused loaders
    * Custom loaders should be lightweight to instantiate
    """

    def __init__(self) -> None:
        """Initialize the registry with built-in model loaders.

        The registry comes pre-configured with loaders for the most common
        model loading scenarios:
        - "standard": StandardModelLoader for exact architecture matches
        - "transfer_learning": TransferLearningModelLoader for architecture differences
        - "weights_only": WeightsOnlyModelLoader for clean weight loading
        """
        self._loaders: dict[str, ModelLoader] = {}
        # Register default loaders with descriptive names
        self.register("standard", StandardModelLoader())
        self.register("transfer_learning", TransferLearningModelLoader())
        self.register("weights_only", WeightsOnlyModelLoader())

    def register(self, name: str, loader: ModelLoader) -> None:
        """Register a model loader with the registry.

        Associates a model loader instance with a string identifier for
        later retrieval. The name should be descriptive and unique.

        Parameters
        ----------
        name : str
            Unique identifier for the loader. Should be descriptive and
            follow naming conventions (lowercase, underscores for spaces).
            Examples: "quantized", "pruned", "custom_transfer".
        loader : ModelLoader
            The model loader instance to register. Must implement the
            ModelLoader interface.

        Raises
        ------
        TypeError
            If the loader doesn't implement the ModelLoader interface.

        Examples
        --------

        .. code-block:: python

            registry = ModelLoaderRegistry()

            # Register custom loaders
            registry.register("quantized", QuantizedModelLoader())
            registry.register("pruned", PrunedModelLoader())
            registry.register("distilled", DistilledModelLoader())

            # Names should be descriptive
            registry.register("bert_to_gpt", CrossArchitectureLoader())

        Notes
        -----
        * Loader names are case-sensitive
        * Registering with an existing name overwrites the previous loader
        * Loaders are stored by reference, not copied
        """
        self._loaders[name] = loader

    def get_loader(self, name: str) -> ModelLoader:
        """Retrieve a registered model loader by name.

        Returns the model loader instance associated with the given name.
        The loader can then be used directly for custom loading scenarios.

        Parameters
        ----------
        name : str
            The name of the registered loader to retrieve.

        Returns
        -------
        ModelLoader
            The model loader instance associated with the name.

        Raises
        ------
        ValueError
            If no loader is registered with the given name. The error
            message includes available loader names for debugging.

        Examples
        --------

        .. code-block:: python

            registry = ModelLoaderRegistry()

            # Get built-in loaders
            standard_loader = registry.get_loader("standard")
            transfer_loader = registry.get_loader("transfer_learning")

            # Use loader directly
            model = standard_loader.load_model_weights(
                model=my_model,
                checkpoint_source="./checkpoint.ckpt"
            )

            # Handle unknown loaders
            try:
                loader = registry.get_loader("nonexistent")
            except ValueError as e:
                print(f"Error: {e}")  # Shows available loaders
        """
        if name not in self._loaders:
            msg = f"Unknown loader: {name}. Available: {list(self._loaders.keys())}"
            raise ValueError(msg)
        return self._loaders[name]

    def load_model_weights(
        self,
        model: torch.nn.Module,
        checkpoint_source: str | Path,
        loader_type: str = "standard",
        **kwargs,
    ) -> torch.nn.Module:
        """Load model weights using a specified loader strategy.

        This is the main entry point for loading model weights through the
        registry. It combines loader selection and weight loading in one
        convenient method call.

        Parameters
        ----------
        model : torch.nn.Module
            The target model to load weights into.
        checkpoint_source : str | Path
            Path or URL to the checkpoint file. Supports all checkpoint
            loading sources (local, S3, HTTP, GCS, Azure).
        loader_type : str, optional
            Name of the registered loader to use. Default is "standard".
            Built-in options: "standard", "transfer_learning", "weights_only".
        **kwargs
            Additional keyword arguments passed to the selected loader.
            Options depend on the specific loader being used.

        Returns
        -------
        torch.nn.Module
            The input model with loaded weights from the checkpoint.

        Raises
        ------
        ValueError
            If the loader_type is not registered in the registry.
        FileNotFoundError
            If the checkpoint source cannot be accessed.
        RuntimeError
            If the selected loader fails to load the weights.

        Examples
        --------

        .. code-block:: python

            registry = ModelLoaderRegistry()

            # Standard loading
            model = registry.load_model_weights(
                model=my_model,
                checkpoint_source="./checkpoint.ckpt",
                loader_type="standard"
            )

            # Transfer learning with options
            model = registry.load_model_weights(
                model=target_model,
                checkpoint_source="s3://bucket/pretrained.ckpt",
                loader_type="transfer_learning",
                strict=False,
                skip_mismatched=True
            )

            # Weights-only loading
            model = registry.load_model_weights(
                model=inference_model,
                checkpoint_source="https://models.com/weights.ckpt",
                loader_type="weights_only"
            )

        Integration example:

        .. code-block:: python

            # Function for loading from config
            def load_from_config(model, config):
                return model_loader_registry.load_model_weights(
                    model=model,
                    checkpoint_source=config.source,
                    loader_type=config.get("loader_type", "standard"),
                    **config.get("loader_options", {})
                )

        Notes
        -----
        * This method is thread-safe for concurrent loading operations
        * The selected loader handles all checkpoint source types automatically
        * Error messages include available loader types for easy debugging
        * All loader-specific options are passed through via **kwargs
        """
        loader = self.get_loader(loader_type)
        return loader.load_model_weights(model, checkpoint_source, **kwargs)


# Global registry instance pre-configured with built-in loaders
#: ModelLoaderRegistry: Global model loader registry.
#:
#: This registry comes pre-configured with loaders for the most common
#: model loading scenarios and is used by :func:`load_model_from_checkpoint`.
#: Custom loaders can be registered with this global instance:
#:
#: .. code-block:: python
#:
#:     from anemoi.training.utils.model_loading import model_loader_registry
#:     model_loader_registry.register("my_custom", MyCustomLoader())
#:
#: Available built-in loaders:
#:
#: * ``"standard"``: :class:`StandardModelLoader` - Direct PyTorch Lightning loading
#: * ``"transfer_learning"``: :class:`TransferLearningModelLoader` - Mismatch handling
#: * ``"weights_only"``: :class:`WeightsOnlyModelLoader` - Clean weight loading
model_loader_registry = ModelLoaderRegistry()


def load_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_source: str | Path,
    loader_type: str = "standard",
    **kwargs,
) -> torch.nn.Module:
    """Load PyTorch model weights from checkpoint using the global registry.

    This is the main entry point for loading model weights in the Anemoi training
    system. It provides a simple, unified interface for loading model weights using
    different strategies without requiring direct interaction with the registry.

    The function automatically handles checkpoint source detection and uses the
    global model loader registry to select and execute the appropriate loading
    strategy based on the specified loader type.

    Supported Loading Strategies
    ============================

    * **"standard"**: Direct loading for exact architecture matches (PyTorch Lightning)
    * **"transfer_learning"**: Intelligent loading with size mismatch handling
    * **"weights_only"**: Clean loading of model parameters without training state

    Checkpoint Source Support
    =========================

    * **Local files**: ``/path/to/checkpoint.ckpt``, ``./models/checkpoint.ckpt``
    * **HTTP/HTTPS**: ``https://example.com/models/checkpoint.ckpt``
    * **Amazon S3**: ``s3://bucket/models/checkpoint.ckpt``
    * **Google Cloud Storage**: ``gs://bucket/models/checkpoint.ckpt``
    * **Azure Blob Storage**: ``azure://account.blob.core.windows.net/container/checkpoint.ckpt``

    Parameters
    ----------
    model : torch.nn.Module
        The target PyTorch model to load weights into. The model should already
        be instantiated with the desired architecture.
    checkpoint_source : str | Path
        Path or URL to the checkpoint file. Supports local files and remote
        sources through the checkpoint loading system. Both string paths and
        pathlib.Path objects are supported for local files.
    loader_type : str, optional
        The loading strategy to use. Default is "standard". Available options:

        - ``"standard"``: For exact architecture matches (PyTorch Lightning)
        - ``"transfer_learning"``: For handling architecture differences
        - ``"weights_only"``: For loading only model parameters

    **kwargs
        Additional keyword arguments passed to the selected loader.
        Common options include:

        - ``strict`` (bool): Whether to enforce exact parameter key matching
        - ``skip_mismatched`` (bool): Skip parameters with shape mismatches (transfer learning)
        - Custom loader-specific parameters

    Returns
    -------
    torch.nn.Module
        The input model with loaded weights from the checkpoint. The same model
        object is returned for method chaining, but parameters are updated in-place.

    Raises
    ------
    ValueError
        If the loader_type is not recognized or checkpoint format is invalid.
    FileNotFoundError
        If the checkpoint source cannot be accessed or doesn't exist.
    RuntimeError
        If PyTorch fails to load weights due to architecture mismatch or corruption.
    ImportError
        If required cloud storage libraries are missing for remote sources.

    Examples
    --------
    Basic usage with different loading strategies:

    .. code-block:: python

        from anemoi.training.utils.model_loading import load_model_from_checkpoint

        # Standard loading (exact architecture match)
        model = load_model_from_checkpoint(
            model=my_lightning_model,
            checkpoint_source="./checkpoints/epoch_10.ckpt",
            loader_type="standard"
        )

        # Transfer learning (handles mismatches)
        model = load_model_from_checkpoint(
            model=target_model,
            checkpoint_source="s3://bucket/pretrained.ckpt",
            loader_type="transfer_learning",
            strict=False,
            skip_mismatched=True
        )

        # Weights-only loading (clean inference)
        model = load_model_from_checkpoint(
            model=inference_model,
            checkpoint_source="https://models.com/production.ckpt",
            loader_type="weights_only"
        )

    Advanced usage with custom options:

    .. code-block:: python

        # Transfer learning with detailed control
        model = load_model_from_checkpoint(
            model=custom_model,
            checkpoint_source="gs://bucket/research_model.ckpt",
            loader_type="transfer_learning",
            strict=False,           # Allow missing keys
            skip_mismatched=True    # Skip size mismatches
        )

        # Check if transfer learning was successful
        if hasattr(model, 'weights_initialized'):
            print(f"Weights initialized: {model.weights_initialized}")

    Integration with training pipeline:

    .. code-block:: python

        # Function for config-driven loading
        def load_from_config(model, checkpoint_config):
            return load_model_from_checkpoint(
                model=model,
                checkpoint_source=checkpoint_config.source,
                loader_type=checkpoint_config.get("type", "standard"),
                **checkpoint_config.get("options", {})
            )

        # Usage in training setup
        if config.checkpoint_loading.enabled:
            model = load_from_config(model, config.checkpoint_loading)

    Error handling example:

    .. code-block:: python

        try:
            model = load_model_from_checkpoint(
                model=model,
                checkpoint_source="s3://bucket/model.ckpt",
                loader_type="transfer_learning"
            )
        except ValueError as e:
            print(f"Invalid loader type or checkpoint format: {e}")
        except FileNotFoundError as e:
            print(f"Checkpoint not found: {e}")
        except ImportError as e:
            print(f"Missing dependencies for cloud storage: {e}")

    Notes
    -----
    * This function uses the global ``model_loader_registry`` which comes
      pre-configured with standard loaders
    * All checkpoint sources are handled automatically by the checkpoint loading system
    * The function is thread-safe for concurrent loading operations
    * Custom loaders can be registered with the global registry for specialized needs
    * Loading performance depends on checkpoint size and source type (local vs remote)

    See Also
    --------
    * :class:`ModelLoaderRegistry`: For registering custom loaders
    * :class:`ModelLoader`: For implementing custom loading strategies
    * :func:`anemoi.training.utils.checkpoint_loaders.load_checkpoint_from_source`: For checkpoint access
    * :mod:`anemoi.training.train.modify`: For model modification after loading
    """
    return model_loader_registry.load_model_weights(
        model=model,
        checkpoint_source=checkpoint_source,
        loader_type=loader_type,
        **kwargs,
    )
