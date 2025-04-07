# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import torch
import os
from torch import nn, cuda, Tensor
from typing import Optional
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from contextlib import contextmanager
from torch.utils.checkpoint import checkpoint

from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class CheckpointWrapper(nn.Module):
    """Wrapper for checkpointing a module."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return checkpoint(self.module, *args, **kwargs, use_reentrant=False)


def load_layer_kernels(kernel_config: Optional[DotDict] = {}) -> DotDict:
    """Load layer kernels from the config.

    Args:
        kernel_config : Optional[DotDict]
            Kernel configuration

    Returns:
        DotDict: hydra partial instantiation of the layer kernels
    """
    # If self.layer_kernels entry is missing from the config, use torch.nn kernels
    default_kernels = {
        "Linear": {"_target_": "torch.nn.Linear", "_partial_": True},
        "LayerNorm": {"_target_": "torch.nn.LayerNorm", "_partial_": True},
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



def get_tensor_shape_info(x, str=""):
    """
    Recursively looks for tensors in the input and prints their shape and dtype.
    """
    if isinstance(x, Tensor):
        return str + f"({tuple(x.shape)}, {x.dtype})"
    elif isinstance(x, list):
        str += "["
        for item in x:
            str += get_tensor_shape_info(item) + ", "
        str += "]"
    elif isinstance(x, tuple):
        str += "("
        for item in x:
            str += get_tensor_shape_info(item) + ", "
        str += ")"
    elif isinstance(x, dict):
        str += "{"
        for key, value in x.items():
            str += f"{key}: "
            str += get_tensor_shape_info(value) + ", "
        str += "}"
    else: # add _ for non-tensor items
        str += "_"

    return str

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
        #if(blocking): 
           # torch.cuda.synchronize()
        #  print("CUDA_LAUNCH_BLOCKING= ", os.getenv('CUDA_LAUNCH_BLOCKING'))
        