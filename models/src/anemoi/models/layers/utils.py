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
# should have 1 checkpoitn wrapper per Proc, encoder and decoder

#   The last FW pass shouldnt be sent over just to be sent right back
#   Could accumulate and send over every n layers
#   Small tensors shouldnt be sent over


#TODO
#   Pinned memory isnt being used currently
#   The cost of pinning each tensor might make it not worthwhile https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html#synergies
#   Pinning memory is blocking on the main cpu execution thread
#   Apparently on GH200, it is not required to pin memory


# The streams and min_offload_size come from torchtune https://github.com/pytorch/torchtune/pull/1443/files
# The forward and bw pass skeleton come from unsloth zoo


#class CheckpointWrapper(nn.Module):
    """Wrapper for checkpointing a module."""

    def __init__(self, module: nn.Module, use_pin_memory: bool = True, use_streams: bool = True, min_offload_size: int = 1024, num_layers : Optional[int] = None, offload_every_n_layers: int =  1) -> None:
        super().__init__()

        self.use_pin_memory = use_pin_memory
        self.min_offload_size = min_offload_size

        #used during processor offloading, every n layers is offloaded instead of every layer
        self.offload_every_n_layers = offload_every_n_layers

        #track the number of layers during the proc so we dont offload the final layer just to copy it right back
        if num_layers is not None:
            self.num_layers=num_layers
            self.current_layer=0

        #should check if torch version less then 2.5.0
        if torch.__version__ < "2.5.0.dev20240907" and use_streams:
                use_streams = False
        self.use_streams= use_streams

        self.s0 = torch.cuda.default_stream()  # comp stream
        if self.use_streams:
            self.s1 = torch.cuda.Stream()  # comms stream

        self.module = module
        self.first_bw_pass=True

   def get_num_bytes_tensor(x: torch.Tensor) -> int:
       # get the number of bytes in a tensor, for memory management purposes
       return (
               x.element_size() * x.nelement()
               )  # x.element_size() * x._base_storage().nbytes()

    #def forward(self, *args, **kwargs):
    #    return checkpoint(self.module, *args, **kwargs, use_reentrant=False)

    def is_offload_layer(self) -> bool:
        #checks if this is a layer we should offload on
        #We shouldnt offload on the final processor layer
        #And we should respect 'offload_every_n_layers' when it is set
        is_last_layer=False
        if self.num_layers is not None:
            if self.current_layer + 1 == self.num_layers
                is_last_layer = True

        is_nth_layer = (self.current_layer % self.offload_every_n_layers == 0)

        return is_nth_layer and not is_last_layer

    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, forward_function, hidden_states, *args, **kwargs):
        num_bytes = get_num_bytes_tensor(hidden_states)
        stream = self.s1 if self.use_streams else self.s0
        if num_bytes > self.min_offload_size and self.is_offload_layer():
            LOGGER.debug(f"Copying {num_bytes} bytes from GPU->CPU {'with s1' if self.use_streams else 'with s0'}"
            with torch.cuda.stream(stream):
                saved_hidden_states = hidden_states.to("cpu", non_blocking = True)
                # GPU-> CPU copies need a sync before they're accessed https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html#other-copy-directions-gpu-cpu-cpu-mps
                # torch.cuda.synchronize()
        else:
            LOGGER.debug(f"Tensor of {num_bytes} bytes is too small to be sent to CPU")
        with torch.no_grad():
            output = forward_function(hidden_states, *args, **kwargs)
        if num_bytes > self.min_offload_size:
            ctx.save_for_backward(saved_hidden_states)
        else:
            ctx.save_for_backward(hidden_states)
        ctx.forward_function = forward_function
        ctx.args = args
        self.current_layer += 1
        return output
    pass

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY):
        #Could I do one sync in the first bw pass
        if self.first_bw_pass:
             #Async copies from GPU -> CPU e.g. (FW pass activation offloading) have to be synced before first access
             # We want to minimise the number of times we sync, therefore only sync during the first BW pass (all forwards have been issued)
             torch.cuda.synchronize()
             self.first_bw_pass = False
             LOGGER.debug("First BW pass! syncing...")
        num_bytes = get_num_bytes_tensor(hidden_states)
        stream = self.s1 if self.use_streams else self.s0

        (hidden_states,) = ctx.saved_tensors
        if num_bytes > self.min_offload_size  and self.is_offload_layer() :
            LOGGER.debug(f"Copying {num_bytes} bytes from CPU->GPU {'with s1' if self.use_streams else 'with s0'}"
            with torch.cuda.stream(stream):
                hidden_states = hidden_states.to("cuda:0", non_blocking = True).detach()
                hidden_states.requires_grad_(True)
                # async copies CPU -> GPU without sync are safe #https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html#other-copy-directions-gpu-cpu-cpu-mps
        with torch.enable_grad():
            (output,) = ctx.forward_function(hidden_states, *ctx.args)
        torch.autograd.backward(output, dY)
        self.current_layer -= 1
        return (None, hidden_states.grad,) + (None,)*len(ctx.args)
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
