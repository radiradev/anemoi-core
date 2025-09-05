import warnings
from collections.abc import Generator
from typing import TYPE_CHECKING

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.training.data.refactor import structure as st
from anemoi.training.train.tasks.refactor.base import BaseGraphModule

if TYPE_CHECKING:
    from anemoi.training.data.refactor.structure import NestedTensor


class ForecastingModule(BaseGraphModule):

    def _step(
        self,
        batch: "NestedTensor",
        validation_mode: bool = False,
        apply_processors: bool = True,
    ) -> Generator[tuple[torch.Tensor | None, dict, list], None, None]:
        """Rollout step for the forecaster.

        Will run pre_processors on batch, but not post_processors on predictions.

        Parameters
        ----------
        batch : torch.Tensor
            Batch to use for rollout
        rollout : Optional[int], optional
            Number of times to rollout for, by default None
            If None, will use self.rollout
        training_mode : bool, optional
            Whether in training mode and to calculate the loss, by default True
            If False, loss will be None
        validation_mode : bool, optional
            Whether in validation mode, and to calculate validation metrics, by default False
            If False, metrics will be empty

        Yields
        ------
        Generator[tuple[Union[torch.Tensor, None], dict, list], None, None]
            Loss value, metrics, and predictions (per step)
        """
        # batch = self.allgather_batch(batch)

        if not apply_processors:
            warnings.warn("Skipping processors")
        batch = self.model.normaliser(batch)
        print(f"Normalising batch: {st.to_str(batch, "Batch")}")
        # removed process_batch, only normaliser is supported for now
        # batch = self.process_batch(batch)

        # Delayed scalers need to be initialized after the pre-processors once
        if False:  # self.is_first_step:
            self.define_delayed_scalers()
            self.is_first_step = False

        # input_latlons = self.indexer.get_latlons(batch["input"])  # (G, S=1, B, 2)
        # target_latlons = self.indexer.get_latlons(batch["target"])  # (G, S=1, B, 2)

        # graph = self.graph_editor.update_graph(self.graph_data, input_latlons, target_latlons)

        # prediction at rollout step rollout_step, shape = (bs, latlon, nvar)
        y_pred = self(batch["input"], self.graph_data.clone().to("cuda"))

        # y includes the auxiliary variables, so we must leave those out when computing the loss
        loss = checkpoint(self.loss, y_pred, batch["target"], use_reentrant=False)

        metrics_next = {}
        if validation_mode:
            metrics_next = self.calculate_val_metrics(y_pred, batch["target"], rollout_step=0)

        return loss, metrics_next, y_pred
