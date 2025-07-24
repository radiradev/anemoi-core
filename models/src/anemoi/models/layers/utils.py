# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from hydra.errors import InstantiationException
from hydra.utils import instantiate
import torch
from torch import nn
from torch import cuda
from torch.utils.checkpoint import checkpoint
from contextlib import contextmanager

from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class CheckpointWrapper(nn.Module):
    """Wrapper for checkpointing a module."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return checkpoint(self.module, *args, **kwargs, use_reentrant=False)


def load_layer_kernels(kernel_config: DotDict) -> DotDict:
    """Load layer kernels from the config.

    Args:
        kernel_config : DotDict
            Kernel configuration, e.g. {"Linear": {"_target_": "torch.nn.Linear", "_partial_": True}}

    Returns:
        DotDict: hydra partial instantiation of the layer kernels
    """
    # If self.layer_kernels entry is missing from the config, use torch.nn kernels
    default_kernels = {
        "Linear": {"_target_": "torch.nn.Linear", "_partial_": True},
        "LayerNorm": {"_target_": "torch.nn.LayerNorm", "_partial_": True},
        "QueryNorm": {
            "_target_": "anemoi.models.layers.normalization.AutocastLayerNorm",
            "_partial_": True,
            "bias": False,
        },
        "KeyNorm": {
            "_target_": "anemoi.models.layers.normalization.AutocastLayerNorm",
            "_partial_": True,
            "bias": False,
        },
    }
    layer_kernels = {**default_kernels, **kernel_config}

    # Loop through all kernels in the layer_kernels config entry and try import them
    for kernel in layer_kernels:
        kernel_entry = layer_kernels[kernel]
        try:
            instantiate(kernel_entry)
        except InstantiationException:
            LOGGER.info(
                f"{kernel_entry['_target_']} not found! check your config.model.layer_kernel.{kernel} entry. Maybe your desired kernel is not installed or the import string is incorrect?"
            )
            raise InstantiationException
        else:
            LOGGER.info(f"{kernel} kernel: {kernel_entry}")
    return layer_kernels

class ProfilerWrapper(nn.Module):
    """Wrapper for checkpointing a module."""

    def __init__(self, module: nn.Module, marker: str) -> None:
        super().__init__()
        self.module = module
        #self.marker=module.__class__.__name__
        self.marker=marker
        self.enabled=True
        
        # Register backward hook for profiling backward pass
        #self.register_full_backward_hook(self._backward_hook)
        self.register_full_backward_pre_hook(self._backward_pre_hook)

    def forward(self, *args, **kwargs):
        #print(f"{args=}, {kwargs=}")
        if(self.enabled):
            cuda.nvtx.range_push(self.marker)
        #tracing_marker=marker.split('- ')[1].split(', input')[0]
        with torch.autograd.profiler.record_function("anemoi-"+self.marker):
            out = self.module(*args, **kwargs)
        if(self.enabled):
            cuda.nvtx.range_pop()
        return out
    
    def _backward_pre_hook(self, module, grad_output):
        """Hook function called before the backward pass"""

        #pop any existing ranges
        cuda.nvtx.range_pop()
        cuda.nvtx.range_push(f"{self.marker}_backward")
        
        return grad_output  # Return unchanged gradients

@contextmanager
def nvtx_wrapper(marker, enabled=True, blocking=True):
    if(enabled):
        cuda.nvtx.range_push(marker)
       # if(blocking):
           # torch.cuda.synchronize()
        #if blocking and 'CUDA_LAUNCH_BLOCKING' not in os.environ:
           # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    yield
    if(enabled):
        cuda.nvtx.range_pop()
