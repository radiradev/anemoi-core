# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections import defaultdict
from collections.abc import Generator
from collections.abc import Mapping
from typing import Optional
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from timm.scheduler import CosineLRScheduler
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.interface import AnemoiModelInterface
from anemoi.training.losses.utils import grad_scaler
from anemoi.training.losses.weightedloss import BaseWeightedLoss
from anemoi.training.utils.jsonify import map_config_to_primitives
from anemoi.training.utils.masks import Boolean1DMask
from anemoi.training.utils.masks import NoOutputMask
from anemoi.utils.config import DotDict

from torch.utils.checkpoint import checkpoint
from anemoi.training.distributed.ensemble import gather_ensemble_members
from anemoi.training.utils.inicond import EnsembleInitialConditions

LOGGER = logging.getLogger(__name__)


class GraphForecaster(pl.LightningModule):
    """Graph neural network forecaster for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        interp_data: dict,
        statistics: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : HeteroData
            Graph object
        statistics : dict
            Statistics of the training data
        data_indices : IndexCollection
            Indices of the training data,
        metadata : dict
            Provenance information
        supporting_arrays : dict
            Supporting NumPy arrays to store in the checkpoint

        """
        super().__init__()

        graph_data = graph_data.to(self.device)

        if config.model.get("output_mask", None) is not None:
            self.output_mask = Boolean1DMask(graph_data[config.graph.data][config.model.output_mask])
        else:
            self.output_mask = NoOutputMask()

        self.model = AnemoiModelInterface(
            statistics=statistics,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays | self.output_mask.supporting_arrays,
            graph_data=graph_data,
            interp_data=interp_data,
            config=DotDict(map_config_to_primitives(OmegaConf.to_container(config, resolve=True))),
        )
        self.config = config
        self.data_indices = data_indices

        self.save_hyperparameters()

        self.latlons_data = graph_data[config.graph.data].x
        self.node_weights = self.get_node_weights(config, graph_data)
        self.node_weights = self.output_mask.apply(self.node_weights, dim=0, fill_value=0.0)

        self.logger_enabled = config.diagnostics.log.wandb.enabled or config.diagnostics.log.mlflow.enabled

        variable_scaling = self.get_variable_scaling(config, data_indices)

        self.internal_metric_ranges, self.val_metric_ranges = self.get_val_metric_ranges(config, data_indices)

        # Check if the model is a stretched grid
        if graph_data["hidden"].node_type == "StretchedTriNodes":
            mask_name = config.graph.nodes.hidden.node_builder.mask_attr_name
            limited_area_mask = graph_data[config.graph.data][mask_name].squeeze().bool()
        else:
            limited_area_mask = torch.ones((1,))

        # Kwargs to pass to the loss function
        loss_kwargs = {"node_weights": self.node_weights}
        # Scalars to include in the loss function, must be of form (dim, scalar)
        # Use -1 for the variable dimension, -2 for the latlon dimension
        # Add mask multiplying NaN locations with zero. At this stage at [[1]].
        # Filled after first application of preprocessor. dimension=[-2, -1] (latlon, n_outputs).
        self.scalars = {
            "variable": (-1, variable_scaling),
            "loss_weights_mask": ((-2, -1), torch.ones((1, 1))),
            "limited_area_mask": (2, limited_area_mask),
        }
        self.updated_loss_mask = False

        self.loss = self.get_loss_function(config.training.training_loss, scalars=self.scalars, **loss_kwargs)

        assert isinstance(self.loss, BaseWeightedLoss) and not isinstance(
            self.loss,
            torch.nn.ModuleList,
        ), f"Loss function must be a `BaseWeightedLoss`, not a {type(self.loss).__name__!r}"

        self.metrics = self.get_loss_function(config.training.validation_metrics, scalars=self.scalars, **loss_kwargs)
        if not isinstance(self.metrics, torch.nn.ModuleList):
            self.metrics = torch.nn.ModuleList([self.metrics])

        if config.training.loss_gradient_scaling:
            self.loss.register_full_backward_hook(grad_scaler, prepend=False)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, self.model_comm_group)

    # Future import breaks other type hints TODO Harrison Cook
    @staticmethod
    def get_loss_function(
        config: DictConfig,
        scalars: Union[dict[str, tuple[Union[int, tuple[int, ...], torch.Tensor]]], None] = None,  # noqa: FA100
        **kwargs,
    ) -> Union[BaseWeightedLoss, torch.nn.ModuleList]:  # noqa: FA100
        """Get loss functions from config.

        Can be ModuleList if multiple losses are specified.

        Parameters
        ----------
        config : DictConfig
            Loss function configuration, should include `scalars` if scalars are to be added to the loss function.
        scalars : Union[dict[str, tuple[Union[int, tuple[int, ...], torch.Tensor]]], None], optional
            Scalars which can be added to the loss function. Defaults to None., by default None
            If a scalar is to be added to the loss, ensure it is in `scalars` in the loss config
            E.g.
                If `scalars: ['variable']` is set in the config, and `variable` in `scalars`
                `variable` will be added to the scalar of the loss function.
        kwargs : Any
            Additional arguments to pass to the loss function

        Returns
        -------
        Union[BaseWeightedLoss, torch.nn.ModuleList]
            Loss function, or list of metrics

        Raises
        ------
        TypeError
            If not a subclass of `BaseWeightedLoss`
        ValueError
            If scalar is not found in valid scalars
        """
        config_container = OmegaConf.to_container(config, resolve=False)
        if isinstance(config_container, list):
            return torch.nn.ModuleList(
                [
                    GraphForecaster.get_loss_function(
                        OmegaConf.create(loss_config),
                        scalars=scalars,
                        **kwargs,
                    )
                    for loss_config in config
                ],
            )

        loss_config = OmegaConf.to_container(config, resolve=True)
        scalars_to_include = loss_config.pop("scalars", [])

        # Instantiate the loss function with the loss_init_config
        loss_function = instantiate(loss_config, **kwargs)

        if not isinstance(loss_function, BaseWeightedLoss):
            error_msg = f"Loss must be a subclass of 'BaseWeightedLoss', not {type(loss_function)}"
            raise TypeError(error_msg)

        for key in scalars_to_include:
            if key not in scalars or []:
                error_msg = f"Scalar {key!r} not found in valid scalars: {list(scalars.keys())}"
                raise ValueError(error_msg)
            loss_function.add_scalar(*scalars[key], name=key)

        return loss_function

    def training_weights_for_imputed_variables(
        self,
        batch: torch.Tensor,
    ) -> None:
        """Update the loss weights mask for imputed variables."""
        if "loss_weights_mask" in self.loss.scalar:
            loss_weights_mask = torch.ones((1, 1), device=batch.device)
            found_loss_mask_training = False
            # iterate over all pre-processors and check if they have a loss_mask_training attribute
            for pre_processor in self.model.pre_processors.processors.values():
                if hasattr(pre_processor, "loss_mask_training"):
                    loss_weights_mask = loss_weights_mask * pre_processor.loss_mask_training
                    found_loss_mask_training = True
                # if transform_loss_mask function exists for preprocessor apply it
                if hasattr(pre_processor, "transform_loss_mask") and found_loss_mask_training:
                    loss_weights_mask = pre_processor.transform_loss_mask(loss_weights_mask)
            # update scaler with loss_weights_mask retrieved from preprocessors
            self.loss.update_scalar(scalar=loss_weights_mask.cpu(), name="loss_weights_mask")
            self.scalars["loss_weights_mask"] = ((-2, -1), loss_weights_mask.cpu())

        self.updated_loss_mask = True

    @staticmethod
    def get_val_metric_ranges(config: DictConfig, data_indices: IndexCollection) -> tuple[dict, dict]:

        metric_ranges = defaultdict(list)
        metric_ranges_validation = defaultdict(list)

        for key, idx in data_indices.internal_model.output.name_to_index.items():
            split = key.split("_")
            if len(split) > 1 and split[-1].isdigit():
                # Group metrics for pressure levels (e.g., Q, T, U, V, etc.)
                metric_ranges[f"pl_{split[0]}"].append(idx)
            else:
                metric_ranges[f"sfc_{key}"].append(idx)

            # Specific metrics from hydra to log in logger
            if key in config.training.metrics:
                metric_ranges[key] = [idx]

        # Add the full list of output indices
        metric_ranges["all"] = data_indices.internal_model.output.full.tolist()

        # metric for validation, after postprocessing
        for key, idx in data_indices.model.output.name_to_index.items():
            # Split pressure levels on "_" separator
            split = key.split("_")
            if len(split) > 1 and split[1].isdigit():
                # Create grouped metrics for pressure levels (e.g. Q, T, U, V, etc.) for logger
                metric_ranges_validation[f"pl_{split[0]}"].append(idx)
            else:
                metric_ranges_validation[f"sfc_{key}"].append(idx)
            # Create specific metrics from hydra to log in logger
            if key in config.training.metrics:
                metric_ranges_validation[key] = [idx]

        # Add the full list of output indices
        metric_ranges_validation["all"] = data_indices.model.output.full.tolist()

        return metric_ranges, metric_ranges_validation

    @staticmethod
    def get_variable_scaling(
        config: DictConfig,
        data_indices: IndexCollection,
    ) -> torch.Tensor:
        variable_loss_scaling = (
            np.ones((len(data_indices.internal_data.output.full),), dtype=np.float32)
            * config.training.variable_loss_scaling.default
        )
        pressure_level = instantiate(config.training.pressure_level_scaler)

        LOGGER.info(
            "Pressure level scaling: use scaler %s with slope %.4f and minimum %.2f",
            type(pressure_level).__name__,
            pressure_level.slope,
            pressure_level.minimum,
        )

        for key, idx in data_indices.internal_model.output.name_to_index.items():
            split = key.split("_")
            if len(split) > 1 and split[-1].isdigit():
                # Apply pressure level scaling
                if split[0] in config.training.variable_loss_scaling.pl:
                    variable_loss_scaling[idx] = config.training.variable_loss_scaling.pl[
                        split[0]
                    ] * pressure_level.scaler(
                        int(split[-1]),
                    )
                else:
                    LOGGER.debug("Parameter %s was not scaled.", key)
            else:
                # Apply surface variable scaling
                if key in config.training.variable_loss_scaling.sfc:
                    variable_loss_scaling[idx] = config.training.variable_loss_scaling.sfc[key]
                else:
                    LOGGER.debug("Parameter %s was not scaled.", key)

        return torch.from_numpy(variable_loss_scaling)

    @staticmethod
    def get_node_weights(config: DictConfig, graph_data: HeteroData) -> torch.Tensor:
        node_weighting = instantiate(config.training.node_loss_weights)
        return node_weighting.weights(graph_data)

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

    def advance_input(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        rollout_step: int,
    ) -> torch.Tensor:
        x = x.roll(-1, dims=1)

        # Get prognostic variables
        x[:, -1, :, :, self.data_indices.internal_model.input.prognostic] = y_pred[
            ...,
            self.data_indices.internal_model.output.prognostic,
        ]

        x[:, -1] = self.output_mask.rollout_boundary(x[:, -1], batch[:, -1], self.data_indices)

        # get new "constants" needed for time-varying fields
        x[:, -1, :, :, self.data_indices.internal_model.input.forcing] = batch[
            :,
            self.multi_step + rollout_step,
            :,
            :,
            self.data_indices.internal_data.input.forcing,
        ]
        return x

    def rollout_step(
        self,
        batch: torch.Tensor,
        rollout: Optional[int] = None,  # noqa: FA100
        training_mode: bool = True,
        validation_mode: bool = False,
    ) -> Generator[tuple[Union[torch.Tensor, None], dict, list], None, None]:  # noqa: FA100
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

        Returns
        -------
        None
            None
        """
        # for validation not normalized in-place because remappers cannot be applied in-place
        batch = self.model.pre_processors(batch, in_place=not validation_mode)

        if not self.updated_loss_mask:
            # update loss scalar after first application and initialization of preprocessors
            self.training_weights_for_imputed_variables(batch)

        # start rollout of preprocessed batch
        x = batch[
            :,
            0 : self.multi_step,
            ...,
            self.data_indices.internal_data.input.full,
        ]  # (bs, multi_step, latlon, nvar)
        msg = (
            "Batch length not sufficient for requested multi_step length!"
            f", {batch.shape[1]} !>= {rollout + self.multi_step}"
        )
        assert batch.shape[1] >= rollout + self.multi_step, msg

        for rollout_step in range(rollout or self.rollout):
            # prediction at rollout step rollout_step, shape = (bs, latlon, nvar)
            y_pred = self(x)

            y = batch[:, self.multi_step + rollout_step, ..., self.data_indices.internal_data.output.full]
            # y includes the auxiliary variables, so we must leave those out when computing the loss
            loss = checkpoint(self.loss, y_pred, y, use_reentrant=False) if training_mode else None

            x = self.advance_input(x, y_pred, batch, rollout_step)

            metrics_next = {}
            if validation_mode:
                metrics_next = self.calculate_val_metrics(
                    y_pred,
                    y,
                    rollout_step,
                )
            yield loss, metrics_next, y_pred

    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        del batch_idx
        batch = self.allgather_batch(batch)

        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []

        for loss_next, metrics_next, y_preds_next in self.rollout_step(
            batch,
            rollout=self.rollout,
            training_mode=True,
            validation_mode=validation_mode,
        ):
            loss += loss_next
            metrics.update(metrics_next)
            y_preds.extend(y_preds_next)

        loss *= 1.0 / self.rollout
        return loss, metrics, y_preds

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
        metrics = {}
        y_postprocessed = self.model.post_processors(y, in_place=False)
        y_pred_postprocessed = self.model.post_processors(y_pred, in_place=False)

        for metric in self.metrics:
            metric_name = getattr(metric, "name", metric.__class__.__name__.lower())

            if not isinstance(metric, BaseWeightedLoss):
                # If not a weighted loss, we cannot feature scale, so call normally
                metrics[f"{metric_name}/{rollout_step + 1}"] = metric(
                    y_pred_postprocessed,
                    y_postprocessed,
                )
                continue

            for mkey, indices in self.val_metric_ranges.items():
                if "scale_validation_metrics" in self.config.training and (
                    mkey in self.config.training.scale_validation_metrics.metrics
                    or "*" in self.config.training.scale_validation_metrics.metrics
                ):
                    with metric.scalar.freeze_state():
                        for key in self.config.training.scale_validation_metrics.scalars_to_apply:
                            metric.add_scalar(*self.scalars[key], name=key)

                        # Use internal model space indices
                        internal_model_indices = self.internal_metric_ranges[mkey]

                        metrics[f"{metric_name}/{mkey}/{rollout_step + 1}"] = metric(
                            y_pred,
                            y,
                            scalar_indices=[..., internal_model_indices],
                        )
                else:
                    if -1 in metric.scalar:
                        exception_msg = (
                            "Validation metrics cannot be scaled over the variable dimension"
                            " in the post processed space. Please specify them in the config"
                            " at `scale_validation_metrics`."
                        )
                        raise ValueError(exception_msg)

                    metrics[f"{metric_name}/{mkey}/{rollout_step + 1}"] = metric(
                        y_pred_postprocessed,
                        y_postprocessed,
                        scalar_indices=[..., indices],
                    )

        return metrics

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        train_loss, _, _ = self._step(batch, batch_idx)
        self.log(
            f"train_{getattr(self.loss, 'name', self.loss.__class__.__name__.lower())}",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.shape[0],
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

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """
        Calculate the loss over a validation batch using the training loss function.

        Parameters
        ----------
        batch : torch.Tensor
            Validation batch
        batch_idx : int
            Batch inces

        Returns
        -------
        None
        """
        with torch.no_grad():
            val_loss, metrics, y_preds = self._step(batch, batch_idx, validation_mode=True)

        self.log(
            f"val_{getattr(self.loss, 'name', self.loss.__class__.__name__.lower())}",
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.shape[0],
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


class GraphEnsForecaster(GraphForecaster):
    """Graph neural network forecaster for ensembles for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        interp_data: dict,
        statistics: dict,
        data_indices: dict,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        statistics : dict
            Statistics of the training data
        data_indices : dict
            Indices of the training data,
        metadata : dict
            Provenance information
        """
        super().__init__(
            config=config,
            graph_data=graph_data,
            interp_data=interp_data,
            statistics=statistics,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        # num_gpus_per_ensemble >= 1 and num_gpus_per_ensemble >= num_gpus_per_model (as per the DDP strategy)
        self.model_comm_group_size = config.hardware.num_gpus_per_model
        assert config.hardware.num_gpus_per_ensemble % config.hardware.num_gpus_per_model == 0, (
            "Invalid ensemble vs. model size GPU group configuration: "
            f"{config.hardware.num_gpus_per_ensemble} mod {config.hardware.num_gpus_per_model} != 0.\
            If you would like to run in deterministic mode, please use aifs-train"
        )

        self.num_gpus_per_model = config.hardware.num_gpus_per_model
        self.num_gpus_per_ensemble = config.hardware.num_gpus_per_ensemble

        self.lr = (
            config.hardware.num_nodes
            * config.hardware.num_gpus_per_node
            * config.training.lr.rate
            / config.hardware.num_gpus_per_ensemble
        )

        LOGGER.info("Base (config) learning rate: %e -- Effective learning rate: %e", config.training.lr.rate, self.lr)

        self.nens_per_device = config.training.ensemble_size_per_device
        self.nens_per_group = (
            config.training.ensemble_size_per_device * self.num_gpus_per_ensemble // config.hardware.num_gpus_per_model
        )
        LOGGER.info("Ensemble size: per device = %d, per ens-group = %d", self.nens_per_device, self.nens_per_group)

        # lazy init ensemble group info, will be set by the DDPEnsGroupStrategy:
        self.ens_comm_group = None
        self.ens_comm_group_id = None
        self.ens_comm_group_rank = None
        self.ens_comm_num_groups = None
        self.ens_comm_group_size = None

        self.ensemble_ic_generator = EnsembleInitialConditions(config=config, data_indices=data_indices)

    def forward(self, x: torch.Tensor, fcstep: int) -> torch.Tensor:
        return self.model(x, fcstep, self.model_comm_group)

    def set_ens_comm_group(
        self,
        ens_comm_group: ProcessGroup,
        ens_comm_group_id: int,
        ens_comm_group_rank: int,
        ens_comm_num_groups: int,
        ens_comm_group_size: int,
    ) -> None:
        self.ens_comm_group = ens_comm_group
        self.ens_comm_group_id = ens_comm_group_id
        self.ens_comm_group_rank = ens_comm_group_rank
        self.ens_comm_num_groups = ens_comm_num_groups
        self.ens_comm_group_size = ens_comm_group_size

    def gather_and_compute_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        loss: torch.nn.Module,
        nens_per_device: int,
        ens_comm_group_size: int,
        ens_comm_group: ProcessGroup,
        model_comm_group: ProcessGroup,
        return_pred_ens: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Gather the ensemble members from all devices in my group.

        Eliminate duplicates (if any) and compute the loss.

        Args:
            y_pred: torch.Tensor
                Predicted state tensor, calculated on self.device
            y: torch.Tensor
                Ground truth
            loss: torch.nn.Module
                Loss function
            nens_per_device: int
                Number of ensemble members per device
            ens_comm_group_size: int
                Size of ensemble communication group
            ens_comm_group: int
                Process ensemble group
            model_comm_group: int
                Process model group
            return_pred_ens: bool
                Validation flag: if True, we return the predicted ensemble (post-gather)

        Returns
        -------
            loss_inc:
                Loss
            y_pred_ens:
                Predictions if validation mode
        """
        y_pred_ens = gather_ensemble_members(
            y_pred,
            dim=1,
            shapes=[y_pred.shape] * ens_comm_group_size,
            nens=nens_per_device,
            ndevices=ens_comm_group_size,
            memspacing=model_comm_group.size(),
            mgroup=ens_comm_group,
            scale_gradients=True,
        )

        # compute the loss
        loss_inc = loss(y_pred_ens, y, squash=True)

        # during validation, we also return the pruned ensemble (from step 2) so we can run diagnostics
        # an explicit cast is needed when running in mixed precision (i.e. with y_pred_ens.dtype == torch.(b)float16)
        return loss_inc, y_pred_ens.to(dtype=y.dtype) if return_pred_ens else None

    def rollout_step(
        self,
        batch: torch.Tensor,
        rollout: Optional[int] = None,  # noqa: FA100
        training_mode: bool = True,
        validation_mode: bool = False,
    ) -> Generator[tuple[Union[torch.Tensor, None], dict, list], None, None]:  # noqa: FA100
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

        Returns
        -------
        None
            None
        """
        # for validation not normalized in-place because remappers cannot be applied in-place

        ens_ic = self.ensemble_ic_generator(batch[0], batch[1] if len(batch) == 2 else None)

        LOGGER.debug("Shapes: batch[0][0].shape = %s, ens_ic.shape = %s", list(batch[0][0].shape), list(ens_ic.shape))

        batch[0] = self.model.pre_processors(batch[0], in_place=not validation_mode)
        x = self.model.pre_processors(ens_ic, in_place=not validation_mode) # not in place required here??? ; shape = (bs, multistep, nens_per_device, latlon, input.full) 

        assert len(x.shape) == 5, f"Expected a 5-dimensional tensor and got {len(x.shape)} dimensions, shape {x.shape}!"
        assert (x.shape[1] == self.multi_step) and (
            x.shape[2] == self.nens_per_device
        ), f"Shape mismatch in x! Expected ({self.multi_step}, {self.nens_per_device}), got ({x.shape[1]}, {x.shape[2]})!"

        msg = (
            "Batch length not sufficient for requested multi_step length!"
            f", {batch[0].shape[1]} !>= {rollout + self.multi_step}"
        )
        assert batch[0].shape[1] >= rollout + self.multi_step, msg

        if not self.updated_loss_mask:
            # update loss scalar after first application and initialization of preprocessors
            self.training_weights_for_imputed_variables(batch[0])

        for rollout_step in range(rollout or self.rollout):
            # prediction at rollout step rollout_step, shape = (bs, latlon, nvar)
            y_pred = self(x, rollout_step)
            y = batch[0][:, self.multi_step + rollout_step, 0, :, self.data_indices.internal_data.output.full] # self.data_indices.data.output.full
            LOGGER.debug("SHAPE: y.shape = %s", list(y.shape))

            # y includes the auxiliary variables, so we must leave those out when computing the loss
            loss, y_pred_ens_group = checkpoint(
                self.gather_and_compute_loss,
                y_pred,
                y,
                self.loss,
                self.nens_per_device,
                self.ens_comm_group_size,
                self.ens_comm_group,
                self.model_comm_group,
                True if validation_mode else False,
                use_reentrant=False,
            ) if training_mode else None

            x = self.advance_input(x, y_pred, batch[0], rollout_step)

            metrics_next = {}
            if validation_mode:
                metrics_next = self.calculate_val_metrics(
                    y_pred_ens_group,
                    y,
                    rollout_step,
                )
            yield loss, metrics_next, y_pred_ens_group if validation_mode else [], ens_ic if validation_mode else None

    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
    ) -> tuple:
        """Training / validation step."""
        del batch_idx

        batch[0] = self.allgather_batch(batch[0])
        if len(batch) == 2:
            batch[1] = self.allgather_batch(batch[1])

        LOGGER.debug(
            "SHAPES: batch[0].shape = %s, batch[1].shape == %s",
            list(batch[0].shape),
            list(batch[1].shape) if len(batch) == 2 else "n/a",
        )

        loss = torch.zeros(1, dtype=batch[0].dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []

        for loss_next, metrics_next, y_preds_next, ens_ic in self.rollout_step(
            batch,
            rollout=self.rollout,
            training_mode=True,
            validation_mode=validation_mode,
        ):
            loss += loss_next
            metrics.update(metrics_next)
            y_preds.extend(y_preds_next)

        loss *= 1.0 / self.rollout
        return loss, metrics, y_preds, ens_ic

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor | dict:
        """Run one training step.

        Args:
            batch: tuple
                Batch data. tuple of length 1 or 2.
                batch[0]: analysis, shape (bs, multi_step + rollout, nvar, latlon)
                batch[1] (optional with ensemble): EDA perturbations, shape (multi_step, nens_per_device, nvar, latlon)
            batch_idx: int
                Training batch index

        Returns
        -------
            train_loss:
                Training loss
        """
        train_loss, _, y_preds, _ = self._step(batch, batch_idx)

        self.log(
            "train_" + self.loss.name,
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch[0].shape[0],
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

    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> None:
        """Run one validation step.

        Args:
            batch: tuple
                Batch data. tuple of length 1 or 2.
                batch[0]: analysis, shape (bs, multi_step + rollout, nvar, latlon)
                batch[1] (optional): EDA perturbations, shape (nens_per_device, multi_step, nvar, latlon)
            batch_idx: int
                Validation batch index
        return (*super().validation_step(batch, batch_idx), self.ens_ic)
        """
        with torch.no_grad():
            val_loss, metrics, y_preds, ens_ic = self._step(batch, batch_idx, validation_mode=True)
        self.log(
            "val_" + self.loss.name,
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch[0].shape[0],
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
                batch_size=batch[0].shape[0],
                sync_dist=True,
            )

        return val_loss, y_preds, ens_ic
