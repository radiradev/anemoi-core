# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any # ???

import logging
from functools import cached_property
import hydra

from anemoi.training.train.train import AnemoiTrainer
from anemoi.utils.config import DotDict


from anemoi.training.distributed.strategy import DDPEnsGroupStrategy
from anemoi.training.train.forecaster import GraphEnsForecaster
from anemoi.training.data.datamodule import AnemoiEnsDataModule

LOGGER = logging.getLogger(__name__)


class AIFSEnsTrainer(AnemoiTrainer):
    """Utility class for training the ensemble model."""

    def __init__(self, config: DotDict) -> None:
        super().__init__(config=config)

    @cached_property
    def datamodule(self) -> AnemoiEnsDataModule:
        """DataModule instance and DataSets."""
        datamodule = AnemoiEnsDataModule(self.config, self.graph_data)
        self.config.data.num_features = len(datamodule.ds_train.data.variables)
        LOGGER.info("Number of data variables: %s", str(len(datamodule.ds_train.data.variables)))
        LOGGER.debug("Variables: %s", str(datamodule.ds_train.data.variables))
        return datamodule

    @cached_property
    def model(self) -> GraphEnsForecaster:
        """Provide the model instance."""
        kwargs = {
            "config": self.config,
            "data_indices": self.data_indices,
            "graph_data": self.graph_data,
            "interp_data": self.interp_data,
            "metadata": self.metadata,
            "statistics": self.datamodule.statistics,
            "supporting_arrays": self.supporting_arrays,
        }

        forecaster = GraphEnsForecaster

        if self.load_weights_only:
            LOGGER.info("Restoring only model weights from %s", self.last_checkpoint)
            return forecaster.load_from_checkpoint(self.last_checkpoint, **kwargs)
        return forecaster(**kwargs)

    @cached_property
    def strategy(self) -> Any:
        return DDPEnsGroupStrategy(
            self.config.hardware.num_gpus_per_model,
            self.config.hardware.num_gpus_per_ensemble,
            self.config.dataloader.get("read_group_size", self.config.hardware.num_gpus_per_ensemble),
            static_graph=not self.config.training.accum_grad_batches > 1,
        )


# @hydra.main(version_base=None, config_path="../config", config_name="config")
# def main(config: DotDict) -> None:
#     AnemoiTrainer(config).train()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def ensemble(config: DotDict) -> None:
    AIFSEnsTrainer(config).train()


if __name__ == "__main__":
    ensemble()
