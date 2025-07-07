# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import uuid
from typing import Optional

import torch
from hydra.utils import instantiate
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.models.mult_encoder_processor_decoder import AnemoiMultiModel
from anemoi.models.preprocessing import Processors
from anemoi.training.data.refactor.draft import SampleProvider
from anemoi.utils.config import DotDict


class AnemoiModelInterface(torch.nn.Module):
    """An interface for Anemoi models.

    This class is a wrapper around the Anemoi model that includes pre-processing and post-processing steps.
    It inherits from the PyTorch Module class.

    Attributes
    ----------
    config : DotDict
        Configuration settings for the model.
    id : str
        A unique identifier for the model instance.
    multi_step : bool
        Whether the model uses multi-step input.
    graph_data : HeteroData
        Graph data for the model.
    statistics : dict
        Statistics for the data.
    metadata : dict
        Metadata for the model.
    supporting_arrays : dict
        Numpy arraysto store in the checkpoint.
    data_indices : dict
        Indices for the data.
    pre_processors : Processors
        Pre-processing steps to apply to the data before passing it to the model.
    post_processors : Processors
        Post-processing steps to apply to the model's output.
    model : AnemoiModelEncProcDec
        The underlying Anemoi model.
    """

    def __init__(
        self,
        *,
        config: DotDict,
        sample_provider: SampleProvider,
        graph_data: HeteroData,
        # data_indices: dict,
        metadata: dict,
    ) -> None:
        super().__init__()
        self.config = config
        self.id = str(uuid.uuid4())
        self.sample_provider = sample_provider
        self.graph_data = graph_data
        self.metadata = metadata
        self.supporting_arrays = {}
        # self.data_indices = data_indices
        self._build_model()

    def _build_model(self) -> None:
        """Builds the model and pre- and post-processors."""
        # Instantiate processors
        preprocessors = self.sample_provider.processors(0)
        input_preprocessors = preprocessors["input"]
        target_processors = preprocessors["target"]

        # Assign the processor list pre- and post-processors
        self.input_pre_processors = Processors({}) # Processors(input_preprocessors)
        self.target_pre_processors = Processors({}) # Processors(target_processors)
        self.target_post_processors = Processors({}) # Processors(target_processors, inverse=True)

        # Instantiate the model
        self.model = AnemoiMultiModel(
            # self.config.model.model,
            model_config=self.config,
            sample_provider=self.sample_provider,
            # data_indices=self.data_indices,
            graph_data=self.graph_data,
            # truncation_data=self.truncation_data,
            # _recursive_=False,  # Disables recursive instantiation by Hydra
        )

        # Use the forward method of the model directly
        self.forward = self.model.forward

    def predict_step(
        self, batch: torch.Tensor, model_comm_group: Optional[ProcessGroup] = None, **kwargs
    ) -> torch.Tensor:
        """Prediction step for the model.

        Parameters
        ----------
        batch : torch.Tensor
            Input batched data.

        Returns
        -------
        torch.Tensor
            Predicted data.
        """
        batch = self.pre_processors(batch, in_place=False)

        with torch.no_grad():

            assert (
                len(batch.shape) == 4
            ), f"The input tensor has an incorrect shape: expected a 4-dimensional tensor, got {batch.shape}!"
            # Dimensions are
            # batch, timesteps, horizonal space, variables
            x = batch[:, 0 : self.multi_step, None, ...]  # add dummy ensemble dimension as 3rd index

            y_hat = self(x, model_comm_group=model_comm_group, **kwargs)

        return self.post_processors(y_hat, in_place=False)
