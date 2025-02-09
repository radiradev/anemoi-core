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

#class SmartTensor(torch.Tensor):
  #  @staticmethod
  #  def __new__(cls, data, pinned=False, *args, **kwargs):
  #      if pinned:
  #          data = data.pin_memory()  # Allocate in pinned memory
  #      return super().__new__(cls, data, *args, **kwargs)

    #def __init__(self, data, pinned=False, **kwargs):
    #    self._pinned=pinned

    #@classmethod
    #def __torch_function__(cls, func, types, args=(), kwargs=None):
        # NOTE: Logging calls Tensor.__repr__, so we can't log __repr__ without infinite recursion
        #if func is not torch.Tensor.__repr__:
            #logging.info(f"func: {func.__name__}, args: {args!r}, kwargs: {kwargs!r}")
        #if func is not torch.Tensor.__repr__:
        #    print(f"func: {func.__name__}, args: {args!r}, kwargs: {kwargs!r}")
        #device_args=()
        #for arg in args:
         #   if hasattr(arg, '_pinned') and arg._pinned:
         #       gpu_temp=arg.to("cuda")
         #       #result = super().__torch_function__(func, types, args, kwargs)
         #       device_args.push(gpu_temp)
         #       #gpu_temp=None
         #       #del gpu_temp 
         #   else:
         #       device_args.push(args)

        #if kwargs is None:
        #    kwargs = {}
        #result = super().__torch_function__(func, types, device_args, kwargs)
        #for darg in device_args:
        #    darg = None
        #return result

#https://pytorch.org/docs/stable/notes/extending.html#extending-torch-with-a-tensor-wrapper-type
#class PinnedTensor(object):
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
            data = data.to("cpu").pin_memory()  # Allocate in pinned memory
        return super().__new__(cls, data, *args, **kwargs)

    def __init__(self, data, pinned=True, **kwargs):
        self._pinned = pinned
        #if self._pinned:
        #    #self._t = torch.tensor(data.pin_memory(), device="cpu", pin_memory=True, **kwargs)
        #    self._t = torch.as_tensor(data.to("cpu").pin_memory(), **kwargs)
        #else:
        #    self._t = torch.tensor(data, **kwargs)

    #def __getattr__(self, name):
    #    """Delegate attribute access to the underlying tensor."""
    #    return getattr(self._t, name)

    #def __repr__(self):
    #    return f"PinnedTensor({self._t})"

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

    gpu_data=torch.randn(5, 5, device="cuda")
    result = pinned_tensor + gpu_data

    print(f"before: {pinned_tensor=}, {gpu_data=}. \nafter: {result=}") #demonstrates a cpu tensor being automatically moved to cuda and the result kept there

    import einops
    einops.rearrange(pinned_tensor, "batch time -> (batch time)")

#demo()
