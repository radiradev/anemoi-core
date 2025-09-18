# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from abc import abstractmethod
from collections import defaultdict
from typing import Optional

import torch
from torch import Tensor
from torch import nn

from anemoi.models.data_indices.collection import IndexCollection

LOGGER = logging.getLogger(__name__)


class BasePreprocessor(nn.Module):
    """Base class for data pre- and post-processors."""

    def __init__(
        self,
        config=None,
        dataset: Optional = None,
    ) -> None:
        """Initialize the preprocessor.

        Parameters
        ----------
        config : DotDict
            configuration object of the processor
        data_indices : IndexCollection
            Data indices for input and output variables
        statistics : dict
            Data statistics dictionary

        Attributes
        ----------
        default : str
            Default method for variables not specified in the config
        method_config : dict
            Dictionary of the methods with lists of variables
        methods : dict
            Dictionary of the variables with methods
        data_indices : IndexCollection
            Data indices for input and output variables
        remap : dict
            Dictionary of the variables with remapped names in the config
        """

        super().__init__()

        self.default, self.remap, self.normalizer, self.method_config = self._process_config(config)
        self.methods = _invert_preprocessor_config(self.method_config)

        self.dataset = dataset

    def forward(self, x, in_place: bool = True, inverse: bool = False) -> Tensor:
        """Process the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        in_place : bool
            Whether to process the tensor in place
        inverse : bool
            Whether to inverse transform the input

        Returns
        -------
        torch.Tensor
            Processed tensor
        """
        if not in_place:
            x = x.clone()
        if inverse:
            return self.inverse_transform(x)
        return self.transform(x)

    @abstractmethod
    def transform(self, x, in_place: bool = True) -> Tensor:
        """Process the input tensor."""
        pass

    def inverse_transform(self, x, in_place: bool = True) -> Tensor:
        """Inverse process the input tensor."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement inverse_transform.")


class _Processors(nn.Module):
    """A collection of processors."""

    def __init__(self, processors: list[BasePreprocessor], inverse: bool = False) -> None:
        """Initialize the processors.

        Parameters
        ----------
        processors : list
            List of processors
        """
        super().__init__()

        self.inverse = inverse
        self.first_run = True

        if inverse:
            # Reverse the order of processors for inverse transformation
            # e.g. first impute then normalise forward but denormalise then de-impute for inverse
            processors = processors[::-1]

        self.processors = nn.ModuleDict(processors)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} [{'inverse' if self.inverse else 'forward'}]({self.processors})"

    def forward(self, x, in_place: bool = True) -> Tensor:
        """Process the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        in_place : bool
            Whether to process the tensor in place

        Returns
        -------
        torch.Tensor
            Processed tensor
        """
        for processor in self.processors.values():
            x = processor(x, in_place=in_place, inverse=self.inverse)

        if self.first_run:
            self.first_run = False
            self._run_checks(x)
        return x

    def _run_checks(self, x):
        """Run checks on the processed tensor."""
        if not self.inverse:
            # Forward transformation checks:
            assert not torch.isnan(
                x
            ).any(), f"NaNs ({torch.isnan(x).sum()}) found in processed tensor after {self.__class__.__name__}."


class Processors(nn.Module):
    def __init__(self, processors: dict[str, list[BasePreprocessor]], inverse: bool = False) -> None:
        super().__init__()
        self.dic = nn.ModuleDict({k: _Processors(v, inverse) for k, v in processors.items()})

    def __repr__(self):
        return f"{self.__class__.__name__}({self.dic})"

    def forward(self, x, in_place: bool = True):
        assert isinstance(x, dict), type(x)
        if in_place:
            for k in x.keys():
                x[k] = self.dic[k].forward(x[k], in_place=in_place)
            return x
        else:
            return {k: self.dic[k].forward(x[k], in_place=in_place) for k in x.keys()}
