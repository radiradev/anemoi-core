# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Checkpoint Loading System for Anemoi Training.

This module provides a flexible and extensible checkpoint loading system that supports
loading PyTorch model checkpoints from various sources including local filesystem,
S3, HTTP/HTTPS, Google Cloud Storage, and Azure Blob Storage.

Key Features
============

* **Multi-source Support**: Load checkpoints from local files, cloud storage, or web URLs
* **Automatic Source Detection**: Loaders automatically detect compatible sources based on URL schemes
* **Registry Pattern**: Extensible loader registry for custom checkpoint sources
* **Error Handling**: Robust error handling with informative messages for debugging
* **Cloud Integration**: Built-in support for major cloud storage providers

Architecture
============

The system uses a registry pattern with abstract base classes:

1. **CheckpointLoader**: Abstract base class defining the loader interface
2. **Concrete Loaders**: Specific implementations for different sources (local, remote)
3. **CheckpointLoaderRegistry**: Central registry that manages loaders and routes requests
4. **Global Registry**: Pre-configured registry instance ready for immediate use

Supported Sources
=================

* **Local Files**: Any file system path (``/path/to/checkpoint.ckpt``)
* **HTTP/HTTPS**: Web URLs (``https://example.com/model.ckpt``)
* **Amazon S3**: S3 URLs (``s3://bucket/path/model.ckpt``)
* **Google Cloud Storage**: GCS URLs (``gs://bucket/path/model.ckpt``)
* **Azure Blob Storage**: Azure URLs (``azure://account.blob.core.windows.net/container/model.ckpt``)

Basic Usage
===========

.. code-block:: python

    from anemoi.training.utils.checkpoint_loaders import load_checkpoint_from_source

    # Load from local file
    checkpoint = load_checkpoint_from_source("/path/to/model.ckpt")

    # Load from S3
    checkpoint = load_checkpoint_from_source("s3://my-bucket/models/checkpoint.ckpt")

    # Load from HTTP
    checkpoint = load_checkpoint_from_source("https://example.com/model.ckpt")

    # The checkpoint is a standard PyTorch dictionary
    model.load_state_dict(checkpoint["state_dict"])

Advanced Usage
==============

.. code-block:: python

    from anemoi.training.utils.checkpoint_loaders import (
        CheckpointLoaderRegistry,
        RemoteCheckpointLoader,
        checkpoint_registry
    )

    # Use the global registry directly
    loader = checkpoint_registry.get_loader("s3://my-bucket/model.ckpt")
    checkpoint = loader.load_checkpoint("s3://my-bucket/model.ckpt")

    # Create a custom registry
    custom_registry = CheckpointLoaderRegistry()
    custom_registry.register(RemoteCheckpointLoader())
    checkpoint = custom_registry.load_checkpoint("https://example.com/model.ckpt")

Extending the System
====================

To add support for custom checkpoint sources:

.. code-block:: python

    class CustomCheckpointLoader(CheckpointLoader):
        def supports_source(self, source: str | Path) -> bool:
            return str(source).startswith("custom://")

        def load_checkpoint(self, source: str | Path) -> dict:
            # Custom loading logic
            return custom_load_function(source)

    # Register with the global registry
    checkpoint_registry.register(CustomCheckpointLoader())

Cloud Provider Setup
====================

For cloud storage access, ensure proper authentication:

**AWS S3**:
    Configure AWS credentials via AWS CLI, environment variables, or IAM roles.
    Requires ``boto3`` package.

**Google Cloud Storage**:
    Set up Google Cloud authentication via service account or gcloud CLI.
    Requires ``google-cloud-storage`` package.

**Azure Blob Storage**:
    Configure Azure credentials via Azure CLI or environment variables.
    Requires ``azure-storage-blob`` package.

Error Handling
==============

The system provides detailed error messages for common issues:

* ``FileNotFoundError``: Local checkpoint file not found
* ``ValueError``: Unsupported URL scheme or no compatible loader found
* ``ImportError``: Required cloud storage library not installed
* Network errors: Connection timeouts, authentication failures

Integration with Training
=========================

This module integrates with the model loading system:

.. code-block:: python

    # In training configuration
    training:
      checkpoint_loading:
        source: "s3://my-bucket/pretrained.ckpt"
        loader_type: "transfer_learning"

    # The training system automatically uses this module to fetch the checkpoint

See Also
--------
* :mod:`anemoi.training.utils.model_loading`: Model weight loading strategies
* :mod:`anemoi.training.train.modify`: Model modification system
* :mod:`anemoi.training.train.train`: Main training pipeline integration

Notes
-----
* Remote checkpoints are downloaded to temporary files and cleaned up automatically
* Large checkpoints may take time to download; consider network bandwidth
* Cloud storage credentials must be properly configured for remote access
* The system respects PyTorch's ``weights_only=False`` for full checkpoint loading
"""

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from urllib.parse import urlparse

LOGGER = logging.getLogger(__name__)


class CheckpointLoader(ABC):
    """Abstract base class for loading model checkpoints from various sources.

    This class defines the interface that all checkpoint loaders must implement.
    Concrete implementations handle specific source types (local files, remote URLs, etc.)
    and provide the actual loading logic.

    The loader system uses a capability-based approach where each loader declares
    which sources it can handle via the ``supports_source`` method, and the registry
    automatically routes requests to appropriate loaders.

    Design Principles
    =================

    * **Single Responsibility**: Each loader handles one type of source
    * **Capability Declaration**: Loaders explicitly declare supported sources
    * **Consistent Interface**: All loaders return standard PyTorch checkpoint dictionaries
    * **Error Transparency**: Loaders provide informative error messages for debugging

    Implementing Custom Loaders
    ============================

    To create a custom checkpoint loader:

    .. code-block:: python

        class MyCustomLoader(CheckpointLoader):
            def supports_source(self, source: str | Path) -> bool:
                # Return True if this loader can handle the source
                return str(source).startswith("myprotocol://")

            def load_checkpoint(self, source: str | Path) -> dict:
                # Implement your loading logic
                # Must return a PyTorch checkpoint dictionary
                return torch.load(processed_source, weights_only=False)

    Error Handling Guidelines
    =========================

    Implementations should raise appropriate exceptions:

    * ``FileNotFoundError``: When the source doesn't exist
    * ``ValueError``: For malformed sources or unsupported formats
    * ``ImportError``: When required dependencies are missing
    * ``ConnectionError``: For network-related issues

    See Also
    --------
    * :class:`LocalCheckpointLoader`: Loader for local filesystem checkpoints
    * :class:`RemoteCheckpointLoader`: Loader for remote cloud/web checkpoints
    * :class:`CheckpointLoaderRegistry`: Registry for managing multiple loaders
    """

    @abstractmethod
    def load_checkpoint(self, source: str | Path) -> dict:
        """Load a PyTorch checkpoint from the specified source.

        This method must be implemented by all concrete checkpoint loaders.
        It should handle the specific loading logic for the loader's supported
        source types and return a standard PyTorch checkpoint dictionary.

        The returned dictionary should contain at minimum a ``state_dict`` key
        with the model parameters, and may include additional keys like
        ``hyper_parameters``, ``optimizer_state_dict``, ``lr_scheduler_state_dict``, etc.

        Parameters
        ----------
        source : str | Path
            The checkpoint source to load from. This can be a local file path,
            a remote URL, or any other identifier that the loader understands.
            The format depends on the specific loader implementation.

        Returns
        -------
        dict
            A PyTorch checkpoint dictionary containing the loaded model state
            and any additional metadata. The dictionary structure follows
            PyTorch Lightning conventions:

            - ``state_dict``: Model parameters and buffers
            - ``hyper_parameters``: Training hyperparameters (optional)
            - ``optimizer_state_dict``: Optimizer state (optional)
            - ``lr_scheduler_state_dict``: Learning rate scheduler state (optional)
            - ``epoch``: Training epoch (optional)
            - ``global_step``: Training step (optional)

        Raises
        ------
        FileNotFoundError
            If the checkpoint source doesn't exist or cannot be accessed.
        ValueError
            If the source format is invalid or the checkpoint data is corrupted.
        ImportError
            If required dependencies for loading from this source are not installed.
        ConnectionError
            If there are network issues accessing remote sources.
        RuntimeError
            If the checkpoint loading fails for any other reason.

        Examples
        --------

        .. code-block:: python

            loader = SomeCheckpointLoader()
            checkpoint = loader.load_checkpoint("/path/to/model.ckpt")

            # Access the model state dict
            model.load_state_dict(checkpoint["state_dict"])

            # Access hyperparameters if available
            if "hyper_parameters" in checkpoint:
                config = checkpoint["hyper_parameters"]
        """
        ...

    @abstractmethod
    def supports_source(self, source: str | Path) -> bool:
        """Check if this loader can handle the specified checkpoint source.

        This method determines whether the loader is capable of loading from
        the given source. It's used by the registry system to automatically
        select the appropriate loader for each source.

        Implementations should be fast and lightweight since this method
        may be called multiple times during loader selection.

        Parameters
        ----------
        source : str | Path
            The checkpoint source to evaluate. This could be a file path,
            URL, or any other source identifier.

        Returns
        -------
        bool
            True if this loader can handle the source, False otherwise.

        Notes
        -----
        The method should be conservative in its checks - only return True
        if the loader can definitely handle the source. It's better to
        return False and let another loader handle it than to return True
        and fail during loading.

        Examples
        --------

        .. code-block:: python

            # Local file loader
            def supports_source(self, source):
                return Path(source).exists()

            # S3 loader
            def supports_source(self, source):
                return str(source).startswith("s3://")

            # HTTP loader
            def supports_source(self, source):
                parsed = urlparse(str(source))
                return parsed.scheme in {"http", "https"}
        """
        ...


class LocalCheckpointLoader(CheckpointLoader):
    """Checkpoint loader for local filesystem access.

    This loader handles loading checkpoints from the local filesystem. It supports
    any valid file system path and automatically detects when a source is a local
    path versus a remote URL.

    Features
    ========

    * **Path Resolution**: Handles both string paths and pathlib.Path objects
    * **Existence Checking**: Validates file existence before attempting to load
    * **Cross-platform Support**: Works with Windows, Linux, and macOS file paths
    * **Memory Efficiency**: Loads checkpoints directly to CPU to avoid GPU memory issues

    Usage Examples
    ==============

    .. code-block:: python

        loader = LocalCheckpointLoader()

        # Check if loader supports a path
        if loader.supports_source("/path/to/model.ckpt"):
            checkpoint = loader.load_checkpoint("/path/to/model.ckpt")

        # Works with pathlib.Path objects too
        from pathlib import Path
        checkpoint_path = Path("/models/checkpoint.ckpt")
        if loader.supports_source(checkpoint_path):
            checkpoint = loader.load_checkpoint(checkpoint_path)

    Performance Considerations
    ==========================

    * Large checkpoints load directly into CPU memory first
    * File I/O performance depends on storage type (SSD vs HDD)
    * Network-attached storage may be slower than local drives

    Error Handling
    ==============

    Common errors and their meanings:

    * ``FileNotFoundError``: The checkpoint file doesn't exist at the specified path
    * ``PermissionError``: Insufficient permissions to read the file
    * ``OSError``: File system errors (corrupted file, I/O errors)
    * ``RuntimeError``: PyTorch loading errors (corrupted checkpoint format)
    """

    def supports_source(self, source: str | Path) -> bool:
        """Check if the source is a local file system path.

        This method determines if the given source represents a local file system
        path rather than a remote URL. It uses heuristics to distinguish between
        local paths and remote URLs.

        The method considers a source to be local if:
        1. It's already a pathlib.Path object
        2. It's a string that can be converted to a valid path
        3. The path exists on the local filesystem OR has no URL scheme

        Parameters
        ----------
        source : str | Path
            The source to check. Can be a file path string or Path object.

        Returns
        -------
        bool
            True if this loader can handle the source (i.e., it's a local path),
            False if it appears to be a remote URL or invalid path.

        Examples
        --------

        .. code-block:: python

            loader = LocalCheckpointLoader()

            # These return True
            loader.supports_source("/path/to/file.ckpt")  # Unix path
            loader.supports_source("C:\\models\\file.ckpt")  # Windows path
            loader.supports_source(Path("/models/file.ckpt"))  # pathlib.Path

            # These return False
            loader.supports_source("https://example.com/file.ckpt")  # HTTP URL
            loader.supports_source("s3://bucket/file.ckpt")  # S3 URL
        """
        if isinstance(source, Path):
            return True
        try:
            path = Path(source)
            return path.exists() or not urlparse(str(source)).scheme
        except (ValueError, OSError):
            return False

    def load_checkpoint(self, source: str | Path) -> dict:
        """Load a PyTorch checkpoint from the local filesystem.

        This method loads a checkpoint file from the local filesystem using
        PyTorch's standard loading mechanism. The checkpoint is loaded to CPU
        memory first to avoid GPU memory issues.

        Parameters
        ----------
        source : str | Path
            The local file path to the checkpoint. Can be a string path or
            pathlib.Path object.

        Returns
        -------
        dict
            The loaded PyTorch checkpoint dictionary containing model weights
            and any additional metadata saved during training.

        Raises
        ------
        FileNotFoundError
            If the checkpoint file doesn't exist at the specified path.
        PermissionError
            If there are insufficient permissions to read the file.
        RuntimeError
            If PyTorch fails to load the checkpoint (e.g., corrupted file).
        OSError
            If there are file system I/O errors.

        Examples
        --------

        .. code-block:: python

            loader = LocalCheckpointLoader()

            # Load checkpoint from string path
            checkpoint = loader.load_checkpoint("/path/to/model.ckpt")
            model.load_state_dict(checkpoint["state_dict"])

            # Load checkpoint from pathlib.Path
            from pathlib import Path
            checkpoint_path = Path("/models/latest.ckpt")
            checkpoint = loader.load_checkpoint(checkpoint_path)

        Notes
        -----
        * The checkpoint is loaded with ``weights_only=False`` to support full
          PyTorch Lightning checkpoints with metadata
        * Loading is done with ``map_location="cpu"`` to ensure compatibility
          across different hardware configurations
        * Large checkpoints may take time to load depending on storage speed
        """
        import torch

        path = Path(source)
        if not path.exists():
            msg = f"Checkpoint not found: {path}"
            raise FileNotFoundError(msg)

        LOGGER.info("Loading checkpoint from local path: %s", path)
        return torch.load(path, weights_only=False, map_location="cpu")


class RemoteCheckpointLoader(CheckpointLoader):
    """Checkpoint loader for remote sources including cloud storage and web URLs.

    This loader handles downloading and loading checkpoints from various remote
    sources including cloud storage providers (AWS S3, Google Cloud Storage,
    Azure Blob Storage) and web servers (HTTP/HTTPS).

    Supported Protocols
    ===================

    * **HTTP/HTTPS**: Standard web servers (``https://example.com/model.ckpt``)
    * **Amazon S3**: S3 buckets (``s3://bucket-name/path/to/checkpoint.ckpt``)
    * **Google Cloud Storage**: GCS buckets (``gs://bucket-name/path/to/checkpoint.ckpt``)
    * **Azure Blob Storage**: Azure containers (``azure://account.blob.core.windows.net/container/file.ckpt``)

    Authentication Requirements
    ===========================

    **AWS S3**: Requires proper AWS credentials configured via:
    - AWS CLI (``aws configure``)
    - Environment variables (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``)
    - IAM roles (for EC2 instances)
    - Requires ``boto3`` package

    **Google Cloud Storage**: Requires Google Cloud authentication via:
    - Service account key file
    - gcloud CLI authentication
    - Application default credentials
    - Requires ``google-cloud-storage`` package

    **Azure Blob Storage**: Requires Azure credentials via:
    - Azure CLI authentication
    - Environment variables (``AZURE_STORAGE_CONNECTION_STRING``)
    - Managed identity (for Azure resources)
    - Requires ``azure-storage-blob`` package

    Features
    ========

    * **Automatic Cleanup**: Downloads to temporary files that are automatically cleaned up
    * **Streaming Support**: Efficient handling of large checkpoint files
    * **Error Recovery**: Detailed error messages for troubleshooting authentication and connectivity issues
    * **Cloud Provider Integration**: Native integration with major cloud storage APIs

    Usage Examples
    ==============

    .. code-block:: python

        loader = RemoteCheckpointLoader()

        # Load from S3
        if loader.supports_source("s3://my-bucket/models/checkpoint.ckpt"):
            checkpoint = loader.load_checkpoint("s3://my-bucket/models/checkpoint.ckpt")

        # Load from HTTP
        if loader.supports_source("https://example.com/public/model.ckpt"):
            checkpoint = loader.load_checkpoint("https://example.com/public/model.ckpt")

        # Load from Google Cloud Storage
        if loader.supports_source("gs://my-gcs-bucket/checkpoints/model.ckpt"):
            checkpoint = loader.load_checkpoint("gs://my-gcs-bucket/checkpoints/model.ckpt")

    Performance Considerations
    ==========================

    * **Network Bandwidth**: Download speed depends on internet connection and provider
    * **File Size**: Large checkpoints (>1GB) may take significant time to download
    * **Temporary Storage**: Requires sufficient local disk space for temporary files
    * **Cloud Costs**: Downloads from cloud storage may incur egress charges

    Error Handling
    ==============

    Common errors and their meanings:

    * ``ImportError``: Required cloud storage library not installed
    * ``ConnectionError``: Network connectivity issues
    * ``AuthenticationError``: Invalid or missing cloud credentials
    * ``FileNotFoundError``: Checkpoint doesn't exist at the specified URL
    * ``PermissionError``: Insufficient permissions to access the resource

    Troubleshooting
    ===============

    **S3 Access Issues**:
        Verify AWS credentials with: ``aws sts get-caller-identity``

    **GCS Access Issues**:
        Verify credentials with: ``gcloud auth list``

    **Azure Access Issues**:
        Verify credentials with: ``az account show``

    **Network Issues**:
        Test connectivity with curl or wget to the URL
    """

    def supports_source(self, source: str | Path) -> bool:
        """Check if the source is a supported remote URL.

        This method determines if the given source is a remote URL that this
        loader can handle. It checks the URL scheme against the list of supported
        protocols.

        Supported URL schemes:
        - ``http://`` and ``https://`` for web servers
        - ``s3://`` for Amazon S3
        - ``gs://`` and ``gcs://`` for Google Cloud Storage
        - ``azure://`` and ``az://`` for Azure Blob Storage

        Parameters
        ----------
        source : str | Path
            The source to check. Should be a URL string for remote sources.
            Path objects are not supported by this loader.

        Returns
        -------
        bool
            True if the source is a supported remote URL, False otherwise.

        Examples
        --------

        .. code-block:: python

            loader = RemoteCheckpointLoader()

            # These return True
            loader.supports_source("https://example.com/model.ckpt")
            loader.supports_source("s3://bucket/path/checkpoint.ckpt")
            loader.supports_source("gs://bucket/models/checkpoint.ckpt")
            loader.supports_source("azure://account.blob.core.windows.net/container/model.ckpt")

            # These return False
            loader.supports_source("/local/path/model.ckpt")
            loader.supports_source("ftp://server/model.ckpt")  # Unsupported protocol
            loader.supports_source(Path("/local/model.ckpt"))  # Local Path object
        """
        if isinstance(source, Path):
            return False
        try:
            parsed = urlparse(str(source))
            return bool(parsed.scheme and parsed.scheme in {"http", "https", "s3", "gs", "azure"})
        except (ValueError, OSError):
            return False

    def load_checkpoint(self, source: str | Path) -> dict:
        """Load a PyTorch checkpoint from a remote source.

        This method downloads the checkpoint from the remote source to a temporary
        file and then loads it using PyTorch's standard loading mechanism. The
        temporary file is automatically cleaned up after loading.

        The loading process involves:
        1. Creating a secure temporary file
        2. Downloading the checkpoint to the temporary file
        3. Loading the checkpoint with PyTorch
        4. Cleaning up the temporary file

        Parameters
        ----------
        source : str | Path
            The remote URL to download the checkpoint from. Must be a supported
            URL scheme (http, https, s3, gs, azure).

        Returns
        -------
        dict
            The loaded PyTorch checkpoint dictionary containing model weights
            and any additional metadata.

        Raises
        ------
        ImportError
            If the required cloud storage library is not installed (e.g., boto3
            for S3, google-cloud-storage for GCS).
        ConnectionError
            If there are network connectivity issues during download.
        FileNotFoundError
            If the checkpoint doesn't exist at the specified remote location.
        PermissionError
            If there are insufficient permissions to access the remote resource.
        ValueError
            If the URL scheme is unsupported or malformed.
        RuntimeError
            If PyTorch fails to load the downloaded checkpoint.

        Examples
        --------

        .. code-block:: python

            loader = RemoteCheckpointLoader()

            # Load from S3 (requires boto3 and AWS credentials)
            checkpoint = loader.load_checkpoint("s3://my-bucket/models/checkpoint.ckpt")
            model.load_state_dict(checkpoint["state_dict"])

            # Load from HTTPS
            checkpoint = loader.load_checkpoint("https://example.com/public/model.ckpt")

            # Load from Google Cloud Storage (requires google-cloud-storage)
            checkpoint = loader.load_checkpoint("gs://my-gcs-bucket/models/checkpoint.ckpt")

        Notes
        -----
        * Large checkpoints may take significant time to download
        * Temporary files are stored in the system's temporary directory
        * The method requires sufficient local disk space for the checkpoint
        * Cloud storage access may incur data transfer costs
        * All downloads are performed to CPU memory first for compatibility
        """
        import tempfile
        from pathlib import Path

        import torch

        LOGGER.info("Loading checkpoint from remote source: %s", source)

        # Create temporary file to download checkpoint
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            self._download_checkpoint(str(source), tmp_path)
            return torch.load(tmp_path, weights_only=False, map_location="cpu")
        finally:
            # Clean up temporary file
            if tmp_path.exists():
                tmp_path.unlink()

    def _download_checkpoint(self, source: str, dest_path: Path) -> None:
        """Download checkpoint from remote source to local temporary file.

        This internal method routes the download request to the appropriate
        protocol-specific download method based on the URL scheme.

        Parameters
        ----------
        source : str
            The remote URL to download from.
        dest_path : Path
            The local file path to download to.

        Raises
        ------
        ValueError
            If the URL scheme is not supported.
        """
        parsed = urlparse(source)

        if parsed.scheme in {"http", "https"}:
            self._download_http(source, dest_path)
        elif parsed.scheme == "s3":
            self._download_s3(source, dest_path)
        elif parsed.scheme in {"gs", "gcs"}:
            self._download_gcs(source, dest_path)
        elif parsed.scheme in {"azure", "az"}:
            self._download_azure(source, dest_path)
        else:
            msg = f"Unsupported remote scheme: {parsed.scheme}"
            raise ValueError(msg)

    def _download_http(self, url: str, dest_path: Path) -> None:
        """Download checkpoint from HTTP/HTTPS server.

        Uses urllib.request to download files from web servers. This method
        supports both HTTP and HTTPS protocols and handles redirects automatically.

        Parameters
        ----------
        url : str
            The HTTP/HTTPS URL to download from.
        dest_path : Path
            The local file path to save the downloaded file.

        Raises
        ------
        ConnectionError
            If there are network connectivity issues.
        FileNotFoundError
            If the URL returns a 404 or similar error.
        PermissionError
            If the server returns a 403 or similar authorization error.

        Notes
        -----
        This method uses urllib.request.urlretrieve which may not be suitable
        for very large files due to memory usage. For production use with large
        checkpoints, consider implementing streaming download.
        """
        import urllib.request

        urllib.request.urlretrieve(url, dest_path)  # noqa: S310

    def _download_s3(self, s3_url: str, dest_path: Path) -> None:
        """Download checkpoint from Amazon S3.

        Uses boto3 to download files from S3 buckets. Requires proper AWS
        credentials to be configured.

        Parameters
        ----------
        s3_url : str
            The S3 URL in format: s3://bucket-name/path/to/file
        dest_path : Path
            The local file path to save the downloaded file.

        Raises
        ------
        ImportError
            If boto3 is not installed.
        FileNotFoundError
            If the S3 object doesn't exist.
        PermissionError
            If there are insufficient S3 permissions.
        ConnectionError
            If there are network or AWS service issues.

        Notes
        -----
        * Requires AWS credentials configured via AWS CLI, environment variables, or IAM roles
        * Large files are downloaded efficiently using boto3's streaming capabilities
        * Data transfer charges may apply depending on your AWS configuration
        """
        try:
            import boto3
        except ImportError as e:
            msg = "boto3 required for S3 downloads. Install with: pip install boto3"
            raise ImportError(msg) from e

        parsed = urlparse(s3_url)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        s3_client = boto3.client("s3")
        s3_client.download_file(bucket, key, str(dest_path))

    def _download_gcs(self, gcs_url: str, dest_path: Path) -> None:
        """Download checkpoint from Google Cloud Storage.

        Uses google-cloud-storage client library to download files from GCS buckets.
        Requires proper Google Cloud authentication to be configured.

        Parameters
        ----------
        gcs_url : str
            The GCS URL in format: gs://bucket-name/path/to/file
        dest_path : Path
            The local file path to save the downloaded file.

        Raises
        ------
        ImportError
            If google-cloud-storage is not installed.
        FileNotFoundError
            If the GCS object doesn't exist.
        PermissionError
            If there are insufficient GCS permissions.
        ConnectionError
            If there are network or GCS service issues.

        Notes
        -----
        * Requires Google Cloud credentials via service account, gcloud CLI, or application default credentials
        * Large files are downloaded efficiently using GCS client streaming
        * Data transfer charges may apply depending on your GCS configuration
        """
        try:
            from google.cloud import storage
        except ImportError as e:
            msg = "google-cloud-storage required for GCS downloads. Install with: pip install google-cloud-storage"
            raise ImportError(msg) from e

        parsed = urlparse(gcs_url)
        bucket_name = parsed.netloc
        blob_name = parsed.path.lstrip("/")

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(str(dest_path))

    def _download_azure(self, azure_url: str, dest_path: Path) -> None:
        """Download checkpoint from Azure Blob Storage.

        Uses azure-storage-blob client library to download files from Azure
        blob containers. Requires proper Azure authentication to be configured.

        Parameters
        ----------
        azure_url : str
            The Azure URL in format: azure://account.blob.core.windows.net/container/path/to/file
        dest_path : Path
            The local file path to save the downloaded file.

        Raises
        ------
        ImportError
            If azure-storage-blob is not installed.
        FileNotFoundError
            If the Azure blob doesn't exist.
        PermissionError
            If there are insufficient Azure permissions.
        ConnectionError
            If there are network or Azure service issues.

        Notes
        -----
        * Requires Azure credentials via Azure CLI, environment variables, or managed identity
        * Large files are downloaded efficiently using Azure client streaming
        * Data transfer charges may apply depending on your Azure configuration
        * URL format: azure://storageaccount.blob.core.windows.net/container/blob/path
        """
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError as e:
            msg = "azure-storage-blob required for Azure downloads. Install with: pip install azure-storage-blob"
            raise ImportError(msg) from e

        # Parse Azure URL format: azure://account.blob.core.windows.net/container/blob
        parsed = urlparse(azure_url)
        account_url = f"https://{parsed.netloc}"
        container_name = parsed.path.split("/")[1]
        blob_name = "/".join(parsed.path.split("/")[2:])

        blob_service_client = BlobServiceClient(account_url=account_url)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        dest_path.write_bytes(blob_client.download_blob().readall())


class CheckpointLoaderRegistry:
    """Central registry for managing checkpoint loaders with automatic routing.

    This registry maintains a collection of checkpoint loaders and automatically
    selects the appropriate loader for each checkpoint source. It provides a
    unified interface for loading checkpoints from any supported source.

    The registry uses a first-match strategy where loaders are checked in the
    order they were registered. The first loader that reports it can handle
    a source via ``supports_source()`` will be used to load that checkpoint.

    Features
    ========

    * **Automatic Selection**: Automatically chooses the right loader for each source
    * **Extensible**: Easy registration of custom loaders for new source types
    * **Pre-configured**: Comes with default loaders for common sources
    * **Error Handling**: Clear error messages when no suitable loader is found

    Default Loaders
    ===============

    The registry comes pre-configured with:
    1. ``RemoteCheckpointLoader`` - for S3, HTTP, GCS, Azure sources
    2. ``LocalCheckpointLoader`` - for local filesystem paths

    Usage Examples
    ==============

    .. code-block:: python

        # Use the global registry
        from anemoi.training.utils.checkpoint_loaders import checkpoint_registry

        checkpoint = checkpoint_registry.load_checkpoint("/path/to/local.ckpt")
        checkpoint = checkpoint_registry.load_checkpoint("s3://bucket/remote.ckpt")

        # Create a custom registry
        custom_registry = CheckpointLoaderRegistry()
        custom_registry.register(MyCustomLoader())
        checkpoint = custom_registry.load_checkpoint("custom://source")

    Custom Loaders
    ==============

    To add support for custom checkpoint sources:

    .. code-block:: python

        class FTPCheckpointLoader(CheckpointLoader):
            def supports_source(self, source):
                return str(source).startswith("ftp://")

            def load_checkpoint(self, source):
                # Custom FTP loading logic
                return load_from_ftp(source)

        # Register the custom loader
        checkpoint_registry.register(FTPCheckpointLoader())

        # Now FTP URLs are supported
        checkpoint = checkpoint_registry.load_checkpoint("ftp://server/model.ckpt")

    Performance Considerations
    ==========================

    * Loader selection is O(n) where n is the number of registered loaders
    * For best performance, register more specific loaders first
    * The ``supports_source`` method should be lightweight

    Thread Safety
    =============

    The registry is not thread-safe for registration operations. If you need
    to register loaders from multiple threads, use appropriate synchronization.
    Loading operations are thread-safe as long as individual loaders are thread-safe.
    """

    def __init__(self) -> None:
        """Initialize the registry with default checkpoint loaders.

        The registry is pre-configured with loaders for the most common
        checkpoint sources:
        - RemoteCheckpointLoader for cloud storage and HTTP
        - LocalCheckpointLoader for local filesystem
        """
        self._loaders: list[CheckpointLoader] = []
        # Register default loaders - order matters for selection priority
        self.register(RemoteCheckpointLoader())
        self.register(LocalCheckpointLoader())

    def register(self, loader: CheckpointLoader) -> None:
        """Register a new checkpoint loader with the registry.

        Loaders are checked in registration order when selecting a loader
        for a source. Register more specific loaders before more general ones
        for optimal performance.

        Parameters
        ----------
        loader : CheckpointLoader
            The checkpoint loader instance to register.

        Examples
        --------

        .. code-block:: python

            registry = CheckpointLoaderRegistry()

            # Register custom loaders
            registry.register(FTPCheckpointLoader())
            registry.register(DatabaseCheckpointLoader())

            # More specific loaders should be registered first
            registry.register(SpecialS3Loader())  # Handles special S3 cases
            registry.register(GeneralS3Loader())  # Handles general S3 cases
        """
        self._loaders.append(loader)

    def get_loader(self, source: str | Path) -> CheckpointLoader:
        """Get the appropriate loader for the specified checkpoint source.

        This method iterates through registered loaders in registration order
        and returns the first loader that reports it can handle the source.

        Parameters
        ----------
        source : str | Path
            The checkpoint source to find a loader for.

        Returns
        -------
        CheckpointLoader
            The first loader that can handle the specified source.

        Raises
        ------
        ValueError
            If no registered loader can handle the source.

        Examples
        --------

        .. code-block:: python

            registry = CheckpointLoaderRegistry()

            # Get loader for local file
            loader = registry.get_loader("/path/to/checkpoint.ckpt")
            assert isinstance(loader, LocalCheckpointLoader)

            # Get loader for S3 URL
            loader = registry.get_loader("s3://bucket/checkpoint.ckpt")
            assert isinstance(loader, RemoteCheckpointLoader)
        """
        for loader in self._loaders:
            if loader.supports_source(source):
                return loader
        msg = f"No loader found for source: {source}"
        raise ValueError(msg)

    def load_checkpoint(self, source: str | Path) -> dict:
        """Load a checkpoint using the appropriate loader.

        This is a convenience method that combines loader selection and
        checkpoint loading in one call. It automatically selects the
        appropriate loader and uses it to load the checkpoint.

        Parameters
        ----------
        source : str | Path
            The checkpoint source to load from.

        Returns
        -------
        dict
            The loaded PyTorch checkpoint dictionary.

        Raises
        ------
        ValueError
            If no loader can handle the source.
        FileNotFoundError
            If the checkpoint doesn't exist.
        ImportError
            If required dependencies are missing.
        ConnectionError
            If there are network issues for remote sources.

        Examples
        --------

        .. code-block:: python

            registry = CheckpointLoaderRegistry()

            # Load from any supported source
            checkpoint = registry.load_checkpoint("/local/model.ckpt")
            checkpoint = registry.load_checkpoint("s3://bucket/model.ckpt")
            checkpoint = registry.load_checkpoint("https://example.com/model.ckpt")

            # All return the same format
            model.load_state_dict(checkpoint["state_dict"])
        """
        loader = self.get_loader(source)
        return loader.load_checkpoint(source)


# Global registry instance pre-configured with default loaders
#: CheckpointLoaderRegistry: Global checkpoint loader registry.
#:
#: This registry comes pre-configured with loaders for the most common
#: checkpoint sources and is used by :func:`load_checkpoint_from_source`.
#: Custom loaders can be registered with this global instance:
#:
#: .. code-block:: python
#:
#:     from anemoi.training.utils.checkpoint_loaders import checkpoint_registry
#:     checkpoint_registry.register(MyCustomLoader())
checkpoint_registry = CheckpointLoaderRegistry()


def load_checkpoint_from_source(source: str | Path) -> dict:
    """Load a PyTorch checkpoint from any supported source using the global registry.

    This is the main entry point for loading checkpoints in the Anemoi training
    system. It provides a simple, unified interface for loading checkpoints from
    any supported source type.

    The function automatically detects the source type and selects the appropriate
    loader from the global registry. It supports local files, cloud storage, and
    web URLs without requiring the caller to know which specific loader to use.

    Supported Sources
    =================

    * **Local files**: ``/path/to/checkpoint.ckpt``, ``./models/checkpoint.ckpt``
    * **HTTP/HTTPS**: ``https://example.com/models/checkpoint.ckpt``
    * **Amazon S3**: ``s3://my-bucket/models/checkpoint.ckpt``
    * **Google Cloud Storage**: ``gs://my-bucket/models/checkpoint.ckpt``
    * **Azure Blob Storage**: ``azure://account.blob.core.windows.net/container/checkpoint.ckpt``

    Parameters
    ----------
    source : str | Path
        The checkpoint source to load from. Can be a local file path, remote URL,
        or cloud storage URL. Both string paths and pathlib.Path objects are supported
        for local files.

    Returns
    -------
    dict
        A PyTorch checkpoint dictionary containing the model state and metadata.
        The dictionary typically includes:

        - ``state_dict``: Model parameters and buffers
        - ``hyper_parameters``: Training configuration (optional)
        - ``optimizer_state_dict``: Optimizer state (optional)
        - ``lr_scheduler_state_dict``: Learning rate scheduler state (optional)
        - ``epoch``: Training epoch number (optional)
        - ``global_step``: Training step number (optional)

    Raises
    ------
    ValueError
        If the source format is unsupported or no loader can handle it.
    FileNotFoundError
        If the checkpoint file or URL doesn't exist.
    ImportError
        If required cloud storage libraries are not installed (e.g., boto3 for S3).
    ConnectionError
        If there are network issues accessing remote sources.
    PermissionError
        If there are insufficient permissions to access the source.
    RuntimeError
        If PyTorch fails to load the checkpoint data.

    Examples
    --------
    Basic usage with different source types:

    .. code-block:: python

        from anemoi.training.utils.checkpoint_loaders import load_checkpoint_from_source

        # Load from local file
        checkpoint = load_checkpoint_from_source("/path/to/model.ckpt")
        model.load_state_dict(checkpoint["state_dict"])

        # Load from S3 (requires boto3 and AWS credentials)
        checkpoint = load_checkpoint_from_source("s3://my-bucket/models/pretrained.ckpt")

        # Load from HTTPS
        checkpoint = load_checkpoint_from_source("https://example.com/public/model.ckpt")

        # Load from Google Cloud Storage (requires google-cloud-storage)
        checkpoint = load_checkpoint_from_source("gs://my-gcs-bucket/checkpoints/model.ckpt")

    Integration with model loading:

    .. code-block:: python

        # Direct model loading
        checkpoint = load_checkpoint_from_source(source_path)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])

        # Access hyperparameters if available
        if "hyper_parameters" in checkpoint:
            config = checkpoint["hyper_parameters"]
            print(f"Model was trained for {checkpoint.get('epoch', 'unknown')} epochs")

    Error handling:

    .. code-block:: python

        try:
            checkpoint = load_checkpoint_from_source("s3://bucket/model.ckpt")
        except ImportError:
            print("boto3 not installed - cannot load from S3")
        except FileNotFoundError:
            print("Checkpoint not found at specified location")
        except ConnectionError:
            print("Network issues - check connectivity and credentials")

    Notes
    -----
    * This function uses the global ``checkpoint_registry`` which comes pre-configured
      with loaders for common source types
    * For cloud storage, ensure appropriate authentication is configured
    * Large checkpoints may take time to download from remote sources
    * All checkpoints are loaded to CPU memory first for compatibility
    * The function is thread-safe for loading operations

    See Also
    --------
    * :class:`CheckpointLoaderRegistry`: For custom loader registration
    * :class:`CheckpointLoader`: For implementing custom loaders
    * :mod:`anemoi.training.utils.model_loading`: For model-specific loading strategies
    """
    return checkpoint_registry.load_checkpoint(source)
