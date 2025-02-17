# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional

from hydra.errors import InstantiationException
from hydra.utils import instantiate
from torch import nn
from torch.utils.checkpoint import checkpoint

from anemoi.utils.config import DotDict
import torch
from packaging.version import Version

LOGGER = logging.getLogger(__name__)

torch_version = torch.__version__
if Version(torch_version) < Version("2.4.0"):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type = "cuda")
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type = "cuda")


class CheckpointWrapper(torch.autograd.Function):
#class CheckpointWrapper(nn.Module):
    """Wrapper for checkpointing a module."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    #def forward(self, *args, **kwargs):
    #    return checkpoint(self.module, *args, **kwargs, use_reentrant=False)

    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, forward_function, hidden_states, *args, **kwargs):
        saved_hidden_states = hidden_states.to("cpu", non_blocking = True)
        with torch.no_grad():
            output = forward_function(hidden_states, *args, **kwargs)
        ctx.save_for_backward(saved_hidden_states)
        ctx.forward_function = forward_function
        ctx.args = args
        return output
    pass

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY):
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.to("cuda:0", non_blocking = True).detach()
        hidden_states.requires_grad_(True)
        with torch.enable_grad():
            (output,) = ctx.forward_function(hidden_states, *ctx.args)
        torch.autograd.backward(output, dY)
        return (None, hidden_states.grad,) + (None,)*len(ctx.args)
    pass

class Unsloth_Offloaded_Gradient_Checkpointer(torch.autograd.Function):
    """
    All Unsloth Zoo code licensed under LGPLv3
    Saves VRAM by smartly offloading to RAM.
    Tiny hit to performance, since we mask the movement via non blocking calls.
    """
    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, forward_function, hidden_states, *args):
        saved_hidden_states = hidden_states.to("cpu", non_blocking = True)
        with torch.no_grad():
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(saved_hidden_states)
        ctx.forward_function = forward_function
        ctx.args = args
        return output
    pass

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY):
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.to("cuda:0", non_blocking = True).detach()
        hidden_states.requires_grad_(True)
        with torch.enable_grad():
            (output,) = ctx.forward_function(hidden_states, *ctx.args)
        torch.autograd.backward(output, dY)
        return (None, hidden_states.grad,) + (None,)*len(ctx.args)
    pass
pass


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
