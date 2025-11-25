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

from anemoi.models.preprocessing import Processors
from anemoi.utils.config import DotDict


class AnemoiModelInterface(torch.nn.Module):
    """An interface for Anemoi models."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        pre_processors: Processors,
        post_processors: Processors,
        multi_step: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.pre_processors = pre_processors
        self.post_processors = post_processors
        self.multi_step = multi_step
        self.id = str(uuid.uuid4())
        self.forward = self.model.forward

    def predict_step(
        self, batch: torch.Tensor, model_comm_group: Optional[ProcessGroup] = None, gather_out: bool = True, **kwargs
    ) -> torch.Tensor:
        """Prediction step for the model.

        Parameters
        ----------
        batch : torch.Tensor
            Input batched data.
        model_comm_group : Optional[ProcessGroup], optional
            model communication group, specifies which GPUs work together
        gather_out : bool, optional
            Whether to gather the output, by default True.

        Returns
        -------
        torch.Tensor
            Predicted data.
        """
        # Prepare kwargs for model's predict_step
        predict_kwargs = {
            "batch": batch,
            "pre_processors": self.pre_processors,
            "post_processors": self.post_processors,
            "multi_step": self.multi_step,
            "model_comm_group": model_comm_group,
        }

        # Add tendency processors if they exist
        if hasattr(self, "pre_processors_tendencies"):
            predict_kwargs["pre_processors_tendencies"] = self.pre_processors_tendencies
        if hasattr(self, "post_processors_tendencies"):
            predict_kwargs["post_processors_tendencies"] = self.post_processors_tendencies

        # Delegate to the model's predict_step implementation with processors
        return self.model.predict_step(**predict_kwargs, **kwargs)
