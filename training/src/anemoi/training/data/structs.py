from abc import ABC, abstractmethod
import torch
import numpy as np
import einops

class TorchTensor(torch.Tensor):
    pass

class Thing(ABC):
    # record: dict of tensor
    # stacked record: dict of stacked tensors
    #  

    @abstractmethod
    def get_element_first_dim(self, i): ...

    def __getitem__(self, i,  *args) -> "Thing":
        return self.get_element_first_dim(i)[*args]


class TensorThing(Thing):
    def __init__(self, tensor: TorchTensor) -> None:
        super().__init__()
        self.tensor = tensor

    def __getitem__(self, *args, **kwargs) -> torch.Tensor:
        return self.tensor.__getitem__(*args, **kwargs)


class GroupedThing(Thing):
    def __init__(self, data: dict[str, Thing]) -> None:
        super().__init__()
        self._forward = data

    def get_element_first_dim(self, i) -> Thing:
        return self._forward[i]


class StackedThing(Thing):

    def __init__(self, data: list) -> None:
        super().__init__()
        self.data = data
    
    def get_element_first_dim(self, i) -> Thing:
        return self.data[i]


class AnemoiTensor:
    pass


def anemoi_thing(*args, dim_type: tuple, **kwargs):
    assert len(kwargs) == 0

    if len(args) == 1:
        if isinstance(args[0], np.array):
            return anemoi_thing(torch.from_numpy(args[0]))
        
    if dim_type[0] == "int":
        return StackedThing([anemoi_thing(x, dim_type=dim_type[1:]) for x in args[0]])

    if dim_type[0] == "tensor":
        x = torch.from_numpy(args[0])
        x = einops.rearrange(x, "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")
        return TorchTensor(x)

    if dim_type[0] == "str":
        return GroupedThing({k: anemoi_thing(v, dim_type=dim_type[1:]) for k, v in args[0].items()})
