# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from abc import abstractmethod

import numpy as np
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection

LOGGER = logging.getLogger(__name__)


class BaseVariableLossScaler(ABC):
    """Configurable method converting variable to loss scaling."""

    def __init__(self, scaling_config: DictConfig, data_indices: IndexCollection) -> None:
        """Initialise Scaler.

        Parameters
        ----------
        scaling_config :
        data_indices :

        """
        self.scaling_config = scaling_config
        self.data_indices = data_indices
        self.variable_groups = self.scaling_config.variable_groups
        # turn dictionary around
        self.group_variables = {}
        for group, variables in self.variable_groups.items():
            if isinstance(variables, str):
                variables = [variables]
            for variable in variables:
                self.group_variables[variable] = group
        self.default_group = self.scaling_config.variable_groups.default

    @abstractmethod
    def get_variable_scaling(self) -> np.ndarray: ...

    def get_variable_group(self, variable_name: str) -> tuple[str, str, int]:
        """Get the group of a variable.

        Parameters
        ----------
        variable_name : str
            Name of the variable.

        Returns
        -------
        str
            Group of the variable given in the training-config file.
        str
            Variable reference which corresponds to the variable name without the variable level
        str
            Variable level, i.e. pressure level or model level

        """
        split = variable_name.split("_")
        variable_level = None
        if len(split) > 1 and split[-1].isdigit():
            variable_level = int(split[-1])
            variable_name = variable_name[: -len(split[-1]) - 1]
        if variable_name in self.group_variables:
            return self.group_variables[variable_name], variable_name, variable_level
        return self.default_group, variable_name, variable_level


class GeneralVariableLossScaler(BaseVariableLossScaler):
    """General scaling of variables to loss scaling."""

    def get_variable_scaling(self) -> np.ndarray:
        variable_loss_scaling = (
            np.ones((len(self.data_indices.internal_data.output.full),), dtype=np.float32) * self.scaling_config.default
        )

        for variable_name, idx in self.data_indices.internal_model.output.name_to_index.items():
            _, variable_ref, _ = self.get_variable_group(variable_name)
            # Apply variable scaling by base variable name (variable_ref: variable name without variable level)
            variable_loss_scaling[idx] = self.scaling_config.get(
                variable_ref,
                1.0,
            )
            # TODO(all): do we want to allow scaling by variable_ref and variable_name?
            # i.e. scale q_50 by value for q_50 AND q
            if variable_ref != variable_name:
                variable_loss_scaling[idx] *= self.scaling_config.get(
                    variable_name,
                    1.0,
                )

        return variable_loss_scaling


class BaseVariableLevelScaler(BaseVariableLossScaler):
    """Configurable method converting variable level to scaling."""

    def __init__(
        self,
        scaling_config: DictConfig,
        data_indices: IndexCollection,
        group: str,
        y_intercept: float,
        slope: float,
        name: str,
        scale_dim: int,
    ) -> None:
        """Initialise variable level scaler.

        Parameters
        ----------
        scaling_config : DictConfig
            Configuration for variable loss scaling.
        data_indices : IndexCollection
            Collection of data indices.
        group : str
            Group of variables to scale.
        y_intercept : float
            Y-axis shift of scaling function.
        slope : float
            Slope of scaling function.
        """
        super().__init__(scaling_config, data_indices)
        self.scaling_group = group
        self.y_intercept = y_intercept
        self.slope = slope
        self.name = name
        self.scale_dim = scale_dim

    @abstractmethod
    def get_level_scaling(self, variable_level: int) -> float: ...

    def get_variable_scaling(self) -> np.ndarray:
        variable_level_scaling = np.ones((len(self.data_indices.internal_data.output.full),), dtype=np.float32)

        for variable_name, idx in self.data_indices.internal_model.output.name_to_index.items():
            variable_group, _, variable_level = self.get_variable_group(variable_name)
            if variable_group != self.scaling_group:
                continue
            # Apply variable level scaling
            assert variable_level is not None, f"Variable {variable_name} has no level to scale."
            variable_level_scaling[idx] = self.get_level_scaling(float(variable_level))

        return variable_level_scaling

class LinearVariableLevelScaler(BaseVariableLevelScaler):
    """Linear with slope self.slope, yaxis shift by self.y_intercept."""

    def get_level_scaling(self, variable_level: float) -> np.ndarray:
        return variable_level * self.slope + self.y_intercept


class ReluVariableLevelScaler(BaseVariableLevelScaler):
    """Linear above self.y_intercept, taking constant value self.y_intercept below."""

    def get_level_scaling(self, variable_level: float) -> np.ndarray:
        return max(self.y_intercept, variable_level * self.slope)


class PolynomialVariableLevelScaler(BaseVariableLevelScaler):
    """Polynomial scaling, (slope * variable_level)^2, yaxis shift by self.y_intercept."""

    def get_level_scaling(self, variable_level: float) -> np.ndarray:
        return (self.slope * variable_level) ** 2 + self.y_intercept


class NoVariableLevelScaler(BaseVariableLevelScaler):
    """Constant scaling by 1.0."""

    def __init__(
        self,
        scaling_config: DictConfig,
        data_indices: IndexCollection,
        group: str,
        slope: float = 0.0,
        y_intercept: float = 1.0,
    ) -> None:
        """Initialise Scaler with constant scaling of 1."""
        assert (
            y_intercept == 1.0 and slope == 0
        ), "self.y_intercept must be 1.0 and self.slope 0.0 for no scaling to fit with definition of linear function."
        super().__init__(scaling_config, data_indices, group, slope=0.0, y_intercept=1.0)

    @staticmethod
    def get_level_scaling(variable_level: float) -> np.ndarray:
        del variable_level  # unused
        # no scaling, always return 1.0
        return 1.0


class BaseTendencyScaler(ABC):
    """Configurable method to scale prognostic variables based on data statistics and statistics_tendencies."""

    @abstractmethod
    def scaler(self, variable_stdev: float, variable_tendency_stdev: float) -> float: ...


class NormTendencyScaler(BaseTendencyScaler):
    """Scale loses by stdev of tendency statistics."""
    @staticmethod
    def scaler(variable_stdev: float, variable_tendency_stdev: float) -> float:
        return variable_stdev / variable_tendency_stdev