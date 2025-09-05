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
from typing import Optional

import numpy as np
import torch

import anemoi.training.data.refactor.structure as st
from anemoi.models.preprocessing import BasePreprocessor

LOGGER = logging.getLogger(__name__)


@st.make_output_callable
@st.apply_to_box
def build_normaliser(normaliser, name_to_index, statistics, **kwargs):

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

    f = InputNormaliser(config=normaliser, name_to_index=name_to_index, statistics=statistics)

    def func(data, **kwargs_):

        if not isinstance(data, torch.Tensor):
            raise ValueError(f"Input to InputNormaliser must be a torch.Tensor, got {type(data)}: {data}")

        return {"data": f(data), **kwargs_}

    return func


class InputNormaliser(BasePreprocessor):
    """Normalizes input data with a configurable method."""

    def __init__(
        self,
        config=None,
        name_to_index=None,
        statistics=None,
        dataset: Optional = None,
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
        super().__init__(config, dataset)

        minimum = statistics["minimum"]
        maximum = statistics["maximum"]
        mean = statistics["mean"]
        stdev = statistics["stdev"]

        # Optionally reuse statistic of one variable for another variable
        statistics_remap = {}
        for remap, source in self.remap.items():
            idx_src, idx_remap = name_to_index[source], name_to_index[remap]
            statistics_remap[idx_remap] = (minimum[idx_src], maximum[idx_src], mean[idx_src], stdev[idx_src])

        # Two-step to avoid overwriting the original statistics in the loop (this reduces dependence on order)
        for idx, new_stats in statistics_remap.items():
            minimum[idx], maximum[idx], mean[idx], stdev[idx] = new_stats

        self._validate_normalization_inputs(name_to_index, minimum, maximum, mean, stdev)

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

        # register buffer - this will ensure they get copied to the correct device(s)
        self.register_buffer("_norm_mul", torch.from_numpy(_norm_mul), persistent=True)
        self.register_buffer("_norm_add", torch.from_numpy(_norm_add), persistent=True)

    def _validate_normalization_inputs(self, name_to_index_training_input: dict, minimum, maximum, mean, stdev):
        assert len(self.methods) == sum(len(v) for v in self.method_config.values()), (
            f"Error parsing methods in InputNormaliser methods ({len(self.methods)}) "
            f"and entries in config ({sum(len(v) for v in self.method_config)}) do not match."
        )

        n = minimum.size
        assert maximum.size == n, (maximum.size, n)
        assert mean.size == n, (mean.size, n)
        assert stdev.size == n, (stdev.size, n)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes an input tensor x of shape [..., nvars].

        Normalization done in-place unless specified otherwise.

        The default usecase either assume the full batch tensor or the full input tensor.
        A dataindex is based on the full data can be supplied to choose which variables to normalise.

        Parameters
        ----------
        x : torch.Tensor
            Data to normalize

        Returns
        -------
        torch.Tensor
            _description_
        """

        if self._norm_add.device != x.device or self._norm_mul.device != x.device:
            print(f"Moving normaliser to {x.device}")
            self._norm_add = self._norm_add.to(x.device)
            self._norm_mul = self._norm_mul.to(x.device)

        print(self._norm_mul.shape, self._norm_add.shape)

        x.mul_(self._norm_mul.view(1, -1, 1)).add_(self._norm_add.view(1, -1, 1))
        return x

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalizes an input tensor x of shape [..., nvars | nvars_pred].

        Denormalization done in-place unless specified otherwise.

        The default usecase either assume the full batch tensor or the full output tensor.
        A dataindex is based on the full data can be supplied to choose which variables to denormalise.

        Parameters
        ----------
        x : torch.Tensor
            Data to denormalize

        Returns
        -------
        torch.Tensor
            Denormalized data
        """
        if self._norm_add.device != x.device or self._norm_mul.device != x.device:
            print(f"Moving normaliser to {x.device}")
            self._norm_add = self._norm_add.to(x.device)
            self._norm_mul = self._norm_mul.to(x.device)

        x.subtract_(self._norm_add.view(1, -1, 1)).div_(self._norm_mul.view(1, -1, 1))
        return x
