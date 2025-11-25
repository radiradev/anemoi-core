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
from typing import Any

import pytorch_lightning as pl
import torch

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule

LOGGER = logging.getLogger(__name__)


class AnemoiTrainer(ABC):
    """Utility class for training the model."""

    def __init__(
        self,
        model: pl.LightningModule,
        datamodule: AnemoiDatasetsDataModule,
        callbacks: list,
        loggers: list,
        strategy: pl.strategies.Strategy,
        profiler: pl.profiler.Profiler | None = None,
        # training parameters
        max_epochs: int = 100,
        max_steps: int = -1,
        limit_train_batches: float | int = 1.0,
        limit_val_batches: float | int = 1.0,
        num_sanity_val_steps: int = 2,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float = 0.0,
        gradient_clip_algorithm: str = "norm",
        deterministic: bool = False,
        # hardware parameters
        accelerator: str = "auto",
        devices: int = 1,
        num_nodes: int = 1,
        precision: str = "32-true",
        # diagnostics
        enable_progress_bar: bool = True,
        log_every_n_steps: int = 50,
        anomaly_detection: bool = False,
        print_memory_summary: bool = False,
        check_val_every_n_epoch: int = 1,
        # checkpointing
        last_checkpoint: str | None = None,
        load_weights_only: bool = False,
    ) -> None:
        """Initialize the Anemoi trainer."""
        torch.set_float32_matmul_precision("high")

        self.model = model
        self.datamodule = datamodule
        self.callbacks = callbacks
        self.loggers = loggers
        self.strategy = strategy
        self.profiler = profiler
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.num_sanity_val_steps = num_sanity_val_steps
        self.accumulate_grad_batches = accumulate_grad_batches
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.deterministic = deterministic
        self.accelerator = accelerator
        self.devices = devices
        self.num_nodes = num_nodes
        self.precision = precision
        self.enable_progress_bar = enable_progress_bar
        self.log_every_n_steps = log_every_n_steps
        self.anomaly_detection = anomaly_detection
        self.print_memory_summary = print_memory_summary
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.last_checkpoint = last_checkpoint
        self.load_weights_only = load_weights_only

    def train(self) -> None:
        """Training entry point."""
        LOGGER.debug("Setting up trainer..")

        trainer = pl.Trainer(
            accelerator=self.accelerator,
            callbacks=self.callbacks,
            deterministic=self.deterministic,
            detect_anomaly=self.anomaly_detection,
            strategy=self.strategy,
            devices=self.devices,
            num_nodes=self.num_nodes,
            precision=self.precision,
            max_epochs=self.max_epochs,
            max_steps=self.max_steps,
            logger=self.loggers,
            profiler=self.profiler,
            log_every_n_steps=self.log_every_n_steps,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
            num_sanity_val_steps=self.num_sanity_val_steps,
            accumulate_grad_batches=self.accumulate_grad_batches,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm=self.gradient_clip_algorithm,
            use_distributed_sampler=False,
            enable_progress_bar=self.enable_progress_bar,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
        )

        LOGGER.debug("Starting training..")

        trainer.fit(
            self.model,
            datamodule=self.datamodule,
            ckpt_path=None if (self.load_weights_only) else self.last_checkpoint,
        )

        if self.print_memory_summary:
            LOGGER.info("memory summary: %s", torch.cuda.memory_summary(device=0))

        LOGGER.debug("---- DONE. ----")


def main() -> None:
    """Fiddle-based training entry point."""
    # 1. Configure the pipeline with Fiddle
    #    - This will involve creating fdl.Config objects for all components.
    #    - This configuration will live here, at the application entry point.

    # 2. Instantiate components with fdl.build
    #    - e.g., trainer = fdl.build(trainer_cfg)

    # 3. Run training
    #    - trainer.train()

    # This function will be filled out as other components are refactored
    # to support dependency injection.

    print("New Fiddle-based main function.")


if __name__ == "__main__":
    main()
