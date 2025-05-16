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

    @property
    def keys(self):
        return self._forward.keys()

    def collate(self, other: "GroupedThing") -> "GroupedThing":
        """Collate this GroupedThing with another GroupedThing.
        
        Args:
            other: Another GroupedThing to collate with
            
        Returns:
            A new GroupedThing containing the collated data
        """
        if not isinstance(other, GroupedThing):
            raise TypeError(f"Cannot collate GroupedThing with {type(other)}")
            
        if set(self.keys()) != set(other.keys()):
            raise ValueError("Cannot collate GroupedThings with different keys")
            
        collated_data = {}
        for key in self.keys():
            self_val = self[key]
            other_val = other[key]
            
            if isinstance(self_val, torch.Tensor):
                collated_data[key] = torch.stack([self_val, other_val])
            elif isinstance(self_val, GroupedThing):
                collated_data[key] = self_val.collate(other_val)
            elif isinstance(self_val, StackedThing):
                collated_data[key] = StackedThing(self_val.data + other_val.data)
            else:
                raise TypeError(f"Cannot collate values of type {type(self_val)}")
                
        return GroupedThing(collated_data)


class StackedThing(Thing):

    def __init__(self, data: list) -> None:
        super().__init__()
        self.data = data
    
    def get_element_first_dim(self, i) -> Thing:
        return self.data[i]


class AnemoiTensor:
    pass


def anemoi_thing(*args, dim_type: tuple = None, **kwargs):
    assert dim_type is not None
    assert len(kwargs) == 0

    if len(args) == 1:
        if isinstance(args[0], np.ndarray):
            return anemoi_thing(torch.from_numpy(args[0]), dim_type=dim_type)
        
    if dim_type[0] == "int":
        return StackedThing([anemoi_thing(x, dim_type=dim_type[1:]) for x in args[0]])

    if dim_type[0] == "tensor":
        x = einops.rearrange(args[0], "variables ensemble gridpoints -> ensemble gridpoints variables")
        return TorchTensor(x)

    if dim_type[0] == "str":
        return GroupedThing({k: anemoi_thing(v, dim_type=dim_type[1:]) for k, v in args[0].items()})

def collate_grouped_things(batch: list[GroupedThing]) -> GroupedThing:
    """Collate a batch of GroupedThing objects.
    
    Args:
        batch: List of GroupedThing objects to collate
        
    Returns:
        A single GroupedThing containing the collated data
    """
    if not batch:
        raise ValueError("Cannot collate empty batch")
        
    if not all(isinstance(x, GroupedThing) for x in batch):
        raise TypeError("All elements in batch must be GroupedThing")
        
    result = batch[0]
    for other in batch[1:]:
        result = result.collate(other)
        
    return result
