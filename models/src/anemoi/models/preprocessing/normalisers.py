# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import warnings

import einops
import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def build_normaliser(*args, **kwargs):
    if args and not kwargs:
        if len(args) > 1:
            raise ValueError("Expected at most one positional argument")
        kwargs = args[0]

    if "_target_" not in kwargs:
        # If the normaliser is not a Hydra instantiation, use the config directly
        pass
    elif kwargs.get("_target_") == "anemoi.models.preprocessing.normaliser.InputNormaliser":
        # if the normaliser uses the default class, use the config directly
        pass
    elif kwargs.get("_target_") == "anemoi.models.preprocessing.normalizer.InputNormalizer":
        # backward compatilibily
        # if the normaliser uses the old class with a typo 'z', use the config directly
        pass
    else:
        raise NotImplementedError("TODO: use hydra instanciate for this custom normaliser")

    return Normaliser(**kwargs)


class Normaliser(torch.nn.Module):
    """Normalizes input data with a configurable method."""

    def __init__(
        self,
        normaliser,
        name_to_index,
        statistics,
        dimensions_order,
        _dimensions_order,
        inverse=False,
        **kwargs,
    ) -> None:
        """Initialize the normalizer.

        Parameters
        ----------
        config : DotDict
            configuration object of the processor
        data_indices : IndexCollection
            Data indices for input and output variables
        statistics : dict
            Data statistics dictionary
        """
        super().__init__()

        if dimensions_order != _dimensions_order:
            warnings.warn("❌❌❌ todo fix this dimensions orders mismatch")
        dimensions_order = _dimensions_order

        self.inverse = inverse

        self.methods = self._invert_key_value_list(normaliser)
        self.dimensions_order = dimensions_order

        minimum = statistics["minimum"]
        maximum = statistics["maximum"]
        mean = statistics["mean"]
        stdev = statistics["stdev"]

        _norm_add = np.zeros((minimum.size,), dtype=np.float32)
        _norm_mul = np.ones((minimum.size,), dtype=np.float32)

        for name, i in name_to_index.items():
            method = self.methods.get(name, self.default)

            if method == "mean-std":
                LOGGER.debug(f"Normalizing: {name} is mean-std-normalised.")
                if stdev[i] < (mean[i] * 1e-6):
                    warnings.warn(f"Normalizing: the field seems to have only one value {mean[i]}")
                _norm_mul[i] = 1 / stdev[i]
                _norm_add[i] = -mean[i] / stdev[i]

            elif method == "std":
                LOGGER.debug(f"Normalizing: {name} is std-normalised.")
                if stdev[i] < (mean[i] * 1e-6):
                    warnings.warn(f"Normalizing: the field seems to have only one value {mean[i]}")
                _norm_mul[i] = 1 / stdev[i]
                _norm_add[i] = 0

            elif method == "min-max":
                LOGGER.debug(f"Normalizing: {name} is min-max-normalised to [0, 1].")
                x = maximum[i] - minimum[i]
                if x < 1e-9:
                    warnings.warn(f"Normalizing: the field {name} seems to have only one value {maximum[i]}.")
                _norm_mul[i] = 1 / x
                _norm_add[i] = -minimum[i] / x

            elif method == "max":
                LOGGER.debug(f"Normalizing: {name} is max-normalised to [0, 1].")
                _norm_mul[i] = 1 / maximum[i]

            elif method == "none":
                LOGGER.info(f"Normalizing: {name} is not normalized.")

            else:
                raise ValueError[f"Unknown normalisation method for {name}: {method}"]

        # use self.dimensions_order to know which dimension is the "variables" dimension
        # new_shape is like : "1 variables 1" or "1 1 variables"
        new_shape = ["variables" if d == "variables" else "1" for d in self.dimensions_order]
        new_shape = " ".join([str(d) for d in new_shape])
        _norm_add = einops.rearrange(_norm_add, f"variables -> {new_shape}")
        _norm_mul = einops.rearrange(_norm_mul, f"variables -> {new_shape}")

        # register buffer - this will ensure they get copied to the correct device(s)
        self.register_buffer("_norm_mul", torch.from_numpy(_norm_mul), persistent=True)
        self.register_buffer("_norm_add", torch.from_numpy(_norm_add), persistent=True)

    def forward(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.inverse is False:
            data.mul_(self._norm_mul).add_(self._norm_add)
            return data
        else:
            data.sub_(self._norm_add).div_(self._norm_mul)
            return data

    def _invert_key_value_list(self, method_config: dict[str, list[str]]) -> dict[str, str]:
        self.default = method_config.get("default", "none")
        return {
            variable: method
            for method, variables in method_config.items()
            if not isinstance(variables, str)
            for variable in variables
        }
