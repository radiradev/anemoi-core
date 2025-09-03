# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import reduce
from importlib.util import find_spec

import torch
import torch_geometric
from hydra.utils import get_class
from omegaconf import DictConfig

from anemoi.training.train.tasks.base import BaseGraphModule

LOGGER = logging.getLogger(__name__)


def _get_compile_entry(module: str, compile_config: DictConfig) -> DictConfig | None:
    """Search the compile config for an entry c module name.

    module: str -> full module name e.g. 'anemoi.models.layers.conv.GraphTransformerConv'
    compile_config : DictConfig -> The 'compile' entry within the models config

    returns: None, if 'module' is not listed within 'compile_config'. Otherwise returns the modules entry.

    """
    for entry in compile_config:
        if get_class(entry["module"]) is type(module):
            return entry

    return None


def _meets_library_versions_for_compile() -> bool:
    """Returns True if minimum library versions for compilation in Anemoi is met."""
    has_triton = True
    if find_spec("triton") is None:
        msg = "Triton not installed! Consider installing Triton to "
        msg += "enable compilation and improve speed and memory usage."
        LOGGER.warning(msg)
        has_triton = False

    version_req = torch.__version__ >= "2.6" and torch_geometric.__version__ >= "2.6"

    if not version_req:
        msg = "Minimum library versions for compilation not met. "
        msg += " torch v{torch.__version__} < 2.6 or torch geometric v{torch_geometric.__version__} < 2.6. "
        msg += " Please upgrade these libraries to enable compilation."
        LOGGER.warning(msg)

    return version_req and has_triton


def mark_for_compilation(model: BaseGraphModule, compile_config: DictConfig | None) -> BaseGraphModule:
    """Compiles parts of 'model' according to 'config.model.compile'."""
    if compile_config is None:
        return model

    if not _meets_library_versions_for_compile():
        return model

    LOGGER.info("The following modules will be compiled: %s", str(compile_config))
    default_compile_options = {}

    # Loop through all modules
    for name, module in model.named_modules():
        match = _get_compile_entry(module, compile_config)
        # If it is listed in the compile config
        if match is not None:
            options = match.get("options", default_compile_options)

            LOGGER.debug("%s will be compiled with the following options: %s", str(module), str(options))
            compiled_module = torch.compile(module, **options)  # Note: the module is not compiled yet
            # It is just marked for JIT-compilation later
            # It will be compiled before its first forward pass

            # Update the model with the new 'compiled' module
            # go from "anemoi.models.layers.conv.GraphTransformerConv"
            # to obj(anemoi.models.layers.conv)
            parts = name.split(".")
            parent = reduce(getattr, parts[:-1], model)
            # then set obj(anemoi.models.layers.conv).GrapTransformerConv = compiled_module
            LOGGER.debug("Replacing %s with a compiled version", str(parts[-1]))
            setattr(parent, parts[-1], compiled_module)

    return model
