# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
from abc import ABC
from collections.abc import Generator

import einops
import pytorch_lightning as pl
import torch
from timm.scheduler import CosineLRScheduler
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.interface import AnemoiModelInterface
from anemoi.training.data.refactor.sample_provider import SampleProvider
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.dict import DictLoss
from anemoi.training.losses.scaler_tensor import grad_scaler
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import convert_to_omegaconf
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class BaseGraphModule(pl.LightningModule, ABC):
    """Abstract base class for Anemoi GNN forecasters using PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: BaseSchema,
        graph_data: HeteroData,
        sample_static_info: SampleProvider,
        metadata: dict,
    ) -> None:
        super().__init__()

        self.graph_data = graph_data.to(self.device)

        # TODO: Handle supporting arrays for multiple output masks (multiple outputs)
        # (It is handled in the loss function, but not the version here that is sent to model for supporting_arrays)
        # self.output_mask = instantiate(config.model_dump(by_alias=True).model.output_mask, graph_data=graph_data)

        self.model = AnemoiModelInterface(
            config=convert_to_omegaconf(config),
            sample_static_info=sample_static_info,
            metadata=metadata,
        )

        self.config = config

        # self.save_hyperparameters() # needed for storing the checkpoints

        self.logger_enabled = config.diagnostics.log.wandb.enabled or config.diagnostics.log.mlflow.enabled

        # Instantiate all scalers with the training configuration
        # self.scalers, self.delayed_scaler_builders = create_scalers(
        #    config.model_dump(by_alias=True).training.scalers,
        #    group_config=config.model_dump(by_alias=True).training.variable_groups,
        # data_indices=data_indices,
        #    graph_data=graph_data,
        #    statistics=statistics,
        #    statistics_tendencies=statistics_tendencies,
        #    metadata_variables=metadata["dataset"].get("variables_metadata"),
        #    output_mask=self.output_mask,
        # )

        # self.internal_metric_ranges, self.val_metric_ranges = get_metric_ranges(
        #    config,
        #    data_indices,
        #    metadata["dataset"].get("variables_metadata"),
        # )

        self.loss = get_loss_function(
            config.model_dump(by_alias=True).training.training_loss,
            # scalers={},  # self.scalers,
            #    data_indices=self.data_indices,
        )
        # print_variable_scaling(self.loss, data_indices)

        self.metrics = torch.nn.ModuleDict({})
        #    {
        #        metric_name: get_loss_function(val_metric_config, scalers=self.scalers, data_indices=self.data_indices)
        #        for metric_name, val_metric_config in config.model_dump(
        #            by_alias=True,
        #        ).training.validation_metrics.items()
        #    },
        # )
        if config.training.loss_gradient_scaling:
            self.loss.register_full_backward_hook(grad_scaler, prepend=False)

        self.is_first_step = True
        self.multi_step = config.training.multistep_input
        self.lr = (
            config.hardware.num_nodes
            * config.hardware.num_gpus_per_node
            * config.training.lr.rate
            / config.hardware.num_gpus_per_model
        )
        self.lr_iterations = config.training.lr.iterations
        self.lr_warmup = config.training.lr.warmup
        self.lr_min = config.training.lr.min
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

        self.optimizer_settings = config.training.optimizer

        self.model_comm_group = None
        self.reader_groups = None

        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Rollout increase every : %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)
        LOGGER.debug("Multistep: %d", self.multi_step)

        # lazy init model and reader group info, will be set by the DDPGroupStrategy:
        self.model_comm_group_id = 0
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1

        self.reader_group_id = 0
        self.reader_group_rank = 0

    def forward(self, x: dict[str, torch.Tensor], graph: HeteroData) -> dict[str, torch.Tensor]:
        return self.model(x, graph, model_comm_group=self.model_comm_group)

    def set_model_comm_group(
        self,
        model_comm_group: ProcessGroup,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        model_comm_group_size: int,
    ) -> None:
        self.model_comm_group = model_comm_group
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.model_comm_group_size = model_comm_group_size

    def set_reader_groups(
        self,
        reader_groups: list[ProcessGroup],
        reader_group_id: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        self.reader_groups = reader_groups
        self.reader_group_id = reader_group_id
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

    def allgather_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Allgather the batch-shards across the reader group.

        Parameters
        ----------
        batch : torch.Tensor
            Batch-shard of current reader rank

        Returns
        -------
        torch.Tensor
            Allgathered (full) batch
        """
        grid_size = len(self.latlons_data)  # number of points

        if grid_size == batch.shape[-2]:
            return batch  # already have the full grid

        grid_shard_size = grid_size // self.reader_group_size
        last_grid_shard_size = grid_size - (grid_shard_size * (self.reader_group_size - 1))

        # prepare tensor list with correct shapes for all_gather
        shard_shape = list(batch.shape)
        shard_shape[-2] = grid_shard_size
        last_shard_shape = list(batch.shape)
        last_shard_shape[-2] = last_grid_shard_size

        tensor_list = [torch.empty(tuple(shard_shape), device=self.device) for _ in range(self.reader_group_size - 1)]
        tensor_list.append(torch.empty(last_shard_shape, device=self.device))

        torch.distributed.all_gather(
            tensor_list,
            batch,
            group=self.reader_groups[self.reader_group_id],
        )

        return torch.cat(tensor_list, dim=-2)

    def training_step(
        self,
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        train_loss, _, _ = self._step(batch, batch_idx)
        self.log(
            "train_" + self.loss.name + "_loss",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            # batch_size=batch.shape[0],
            sync_dist=True,
        )
        self.log(
            "rollout",
            float(self.rollout),
            on_step=True,
            logger=self.logger_enabled,
            rank_zero_only=True,
            sync_dist=False,
        )
        return train_loss

    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric: None = None) -> None:
        """Step the learning rate scheduler by Pytorch Lightning.

        Parameters
        ----------
        scheduler : CosineLRScheduler
            Learning rate scheduler object.
        metric : Optional[Any]
            Metric object for e.g. ReduceLRonPlateau. Default is None.

        """
        del metric
        scheduler.step(epoch=self.trainer.global_step)

    def on_train_epoch_end(self) -> None:
        if self.rollout_epoch_increment > 0 and self.current_epoch % self.rollout_epoch_increment == 0:
            self.rollout += 1
            LOGGER.debug("Rollout window length: %d", self.rollout)
        self.rollout = min(self.rollout, self.rollout_max)

    def validation_step(
        self,
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            val_loss, metrics, y_preds = self._step(batch, batch_idx, validation_mode=True)

        self.log(
            "val_" + self.loss.name + "_loss",
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            # batch_size=batch.shape[0],
            sync_dist=True,
        )

        for mname, mvalue in metrics.items():
            self.log(
                "val_" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=self.logger_enabled,
                batch_size=batch.shape[0],
                sync_dist=True,
            )

        return val_loss, y_preds

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict]]:
        """Configure the optimizers and learning rate scheduler.

        Returns
        -------
        tuple[list[torch.optim.Optimizer], list[dict]]
            List of optimizers and list of dictionaries containing the
            learning rate scheduler
        """
        if self.optimizer_settings.zero:
            optimizer = ZeroRedundancyOptimizer(
                self.trainer.model.parameters(),
                lr=self.lr,
                optimizer_class=torch.optim.AdamW,
                **self.optimizer_settings.kwargs,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.trainer.model.parameters(),
                lr=self.lr,
                **self.optimizer_settings.kwargs,
            )

        scheduler = CosineLRScheduler(
            optimizer,
            lr_min=self.lr_min,
            t_initial=self.lr_iterations,
            warmup_t=self.lr_warmup,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def calculate_val_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        rollout_step: int,
    ) -> tuple[dict, list[torch.Tensor]]:
        """Calculate metrics on the validation output.

        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted ensemble
        y: torch.Tensor
            Ground truth (target).
        rollout_step: int
            Rollout step

        Returns
        -------
        val_metrics, preds:
            validation metrics and predictions
        """
        if len(self.metrics) == 0:
            return {}

        metrics = {}
        y_postprocessed = self.model.target_post_processors(y, in_place=False)
        y_pred_postprocessed = self.model.target_post_processors(y_pred, in_place=False)

        for metric_name, metric in self.metrics.items():

            if isinstance(metric, BaseLoss):
                assert isinstance(metric, DictLoss), type(metric)

            if not isinstance(metric, BaseLoss):
                # If not a loss, we cannot feature scale, so call normally
                metrics[f"{metric_name}_metric/{rollout_step + 1}"] = metric(y_pred_postprocessed, y_postprocessed)
                continue

            for mkey, indices in self.val_metric_ranges.items():
                metric_step_name = f"{metric_name}_metric/{mkey}/{rollout_step + 1}"
                if len(metric.scaler.subset_by_dim(TensorDim.VARIABLE.value)):
                    exception_msg = (
                        "Validation metrics cannot be scaled over the variable dimension"
                        " in the post processed space."
                    )
                    raise ValueError(exception_msg)

                metrics[metric_step_name] = metric(y_pred_postprocessed, y_postprocessed, scaler_indices=[..., indices])

        return metrics


class BaseForecasterModule(BaseGraphModule):
    def _step(
        self,
        batch: dict[str, dict[str, torch.Tensor]],
        batch_idx: int,
        validation_mode: bool = False,
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
        del batch_idx
        # batch = self.allgather_batch(batch)

        batch = {
            k: {n: einops.rearrange(t, "bs v t ens xy -> bs t ens xy v") for n, t in v.items()}
            for k, v in batch["data"].items()
        }

        # for validation not normalized in-place because remappers cannot be applied in-place
        # We need shape: (bath_size, time, ens, latlons, n_vars)
        batch["input"] = self.model.model.input_pre_processors(batch["input"], in_place=not validation_mode)
        batch["target"] = self.model.model.target_pre_processors(batch["target"], in_place=not validation_mode)

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
