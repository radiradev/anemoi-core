from collections.abc import Generator
from typing import TYPE_CHECKING

import torch
from torch.utils.checkpoint import checkpoint

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
        # ✅ here, batch = input + target
        # validation vs training, do we have validation batch or training batch?
        #
        # we should create a sample_provider in AnemoiTrainer?
        # or in this module then give it to the dataloader and dataset?

        static_info = self.model.sample_static_info

        batch = static_info.merge_content(batch)
        print(batch.to_str("batch before normalistation"))
        batch = self.model.apply_normalisers(batch)
        print(batch.to_str("batch after normalistation"))

        # print(f"Normalising batch: {batch}")
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
        y_pred = self(batch.input, self.graph_data.clone().to("cuda"))

        target_dtype = next(iter(batch["target"].values()))["data"].dtype
        loss = torch.zeros(1, dtype=target_dtype, device=self.device, requires_grad=True)
        
        # Iterate over all entries in batch["target"] and accumulate loss
        for target_key, target_data in batch["target"].items():
            loss += checkpoint(
                self.loss, y_pred[target_key].unsqueeze(0), # add batch dimension, why do we not get this from the model?
                target_data["data"].permute(0, 2, 1),
                use_reentrant=False
            ) # weighting will probably not be correct here ...
        loss *= 1 / len(batch["target"]) # Average loss over all targets        

        metrics_next = {}
        if validation_mode:
            metrics_next = self.calculate_val_metrics(y_pred, batch["target"], rollout_step=0)

        return loss, metrics_next, y_pred
