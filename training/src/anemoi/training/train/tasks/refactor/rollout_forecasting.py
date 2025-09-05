from typing import TYPE_CHECKING

import torch

from anemoi.training.train.tasks.refactor.forecasting import ForecastingModule

if TYPE_CHECKING:
    from anemoi.training.data.refactor.structure import NestedTensor


class RolloutForecastingModule(ForecastingModule):
    def select_rollout_target(self, batch: dict, rollout_step: int) -> dict:
        return batch["target"][rollout_step]

    def advance_input(
        self,
        x: "NestedTensor",
        y_pred: "NestedTensor",
        batch: "NestedTensor",
        rollout_step: int,
    ) -> dict[str, torch.Tensor]:
        num_target_time = 1  # If f(x_t-1, x_t) = [x_t+1, x_t+2], we should rollout 2 steps instead of 1.
        x = x.roll(-num_target_time, dims=1)  # Roll accross TIME dim

        # Get prognostic variables
        x[-num_target_time:, :, :, self.data_indices.internal_model.input.prognostic] = y_pred[
            ...,
            self.data_indices.internal_model.output.prognostic,
        ]

        # get new "constants" needed for time-varying fields
        x[-num_target_time, :, :, self.data_indices.internal_model.input.forcing] = batch["input"][
            :,
            self.multi_step + rollout_step,
            :,
            :,
            self.data_indices.internal_data.input.forcing,
        ]
        return x

    def _step(self, batch: "NestedTensor", validation_mode: bool = False) -> "NestedTensor":
        batch = self.process_batch(batch)
        x = {"input": batch["input"]}

        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []
        for rollout_step in range(self.rollout):
            x["target"] = self.select_rollout_target(batch, rollout_step)

            loss_next, metrics_next, y_preds_next = super()._step(
                x,
                validation_mode=validation_mode,
                apply_processors=False,
            )

            loss += loss_next
            metrics.update(metrics_next)
            y_preds.append(y_preds_next)

            x["input"] = self.advance_input(x["input"], batch, rollout_step)

        loss *= 1.0 / self.rollout
        return loss, metrics, y_preds
