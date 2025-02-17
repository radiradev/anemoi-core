# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch._C as _C

LOGGER = logging.getLogger(__name__)

#https://pytorch.org/docs/stable/notes/extending.html#extending-torch-with-a-tensor-wrapper-type
class PinnedTensor(torch.Tensor):
    '''
    Extension of the default torch.Tensor
    The tensor will be resident on pinned memory on the CPU.
    When it is operated on with a 'cuda' tensor, a copy of the pinned memory is streamed over
    before being deleted. The result tensor will be stored on device
    This Tensor class is ideal for large, infrequently accessed data (e.g. copies of the input batch at 9km)
    '''

    @staticmethod
    def __new__(cls, data, pinned=False, *args, **kwargs):
        if pinned:
            cpu_data = data.to("cpu").pin_memory()  # Allocate in pinned memory
            data = None #try free whatever the source was
            del data
            return super().__new__(cls, cpu_data, *args, **kwargs)
        else:
            return super().__new__(cls, data, *args, **kwargs)

    def __init__(self, data, pinned=True, **kwargs):
        self._pinned = pinned

    #need this for one of the profiling outputs
    @classmethod
    def new_empty(self, input):
        return PinnedTensor(torch.Tensor(input), pinned=True)


    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        pinneds = tuple(a._pinned for a in args if hasattr(a, '_pinned'))
        #metadatas = tuple(a._metadata for a in args if hasattr(a, '_metadata'))
        #args = [getattr(a, '_t', a) for a in args]

        # you need to guard against every tensor function you invoke in __torch_function__, before you invoke it
        # That includes __repr__ if you want to print
        # If we dont guard against to(), then we recurse infinitely below
        if func == torch.Tensor.to:
            return super().__torch_function__(func, types, args, kwargs)

        processed=[]
        for arg in args:
            if type(arg) == PinnedTensor:
                processed.append(arg.to("cuda", non_blocking=True)) 
            else:
                processed.append(arg)

        return super().__torch_function__(func, types, processed, kwargs)

def demo():
    print("An example of 'PinnedTensor(torch.randn(5, 5)) + torch.randn(5, 5, device='cuda')'")
    #data = torch.randn(5, 5, device="cuda")  # Random tensor
    data = torch.randn(5, 5)  # Random tensor
    pinned_tensor = PinnedTensor(data, pinned=True)
    print(pinned_tensor.shape) #demonstrates that tensor attributes are passed off to the underlying tensor

    pinned_tensor.new_empty([])

    gpu_data=torch.randn(5, 5, device="cuda")
    result = pinned_tensor + gpu_data

    print(f"before: {pinned_tensor=}, {gpu_data=}. \nafter: {result=}") #demonstrates a cpu tensor being automatically moved to cuda and the result kept there

    import einops
    einops.rearrange(pinned_tensor, "batch time -> (batch time)")

#demo()
