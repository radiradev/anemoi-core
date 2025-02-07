# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import matplotlib.cm as cm
from matplotlib.colors import Colormap
from matplotlib.colors import ListedColormap


class CustomColormap(ABC):

    def __init__(self, variables: list | None = None) -> None:
        super().__init__()
        if variables is None:
            variables = []
        self.variables = variables

    @abstractmethod
    def get_cmap(self) -> Colormap: ...


class MatplotlibColormap(CustomColormap):
    def __init__(self, name: str, variables: list | None = None) -> None:
        super().__init__(variables)
        self.name = name
        self.colormap = cm.get_cmap(self.name)

    def get_cmap(self) -> cm:
        return self.colormap


class MatplotlibColormapClevels(CustomColormap):
    def __init__(self, clevels: float, variables: list | None = None) -> None:
        super().__init__(variables)
        self.clevels = clevels
        self.colormap = ListedColormap(self.clevels)

    def get_cmap(self) -> ListedColormap:
        return self.colormap


class DistinctpyColormap(CustomColormap):
    def __init__(self, n_colors: int, variables: list | None = None, colorblind_type: str | None = None) -> None:
        try:
            from distinctipy import distinctipy
        except ImportError:
            error_message = "distinctipy package is not available. Please install it to use DistinctpyColormapN."
            raise ImportError(error_message) from None
        super().__init__(variables)
        self.n_colors = n_colors
        self.colormap = distinctipy.get_colormap(distinctipy.get_colors(n_colors, colorblind_type=colorblind_type))

    def get_cmap(self) -> cm:
        return self.colormap
