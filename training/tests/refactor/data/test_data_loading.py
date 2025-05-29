import pytest
import torch
from omegaconf import DictConfig

from anemoi.training.data.datamodule import AnemoiMultipleDatasetsDataModule


def test_datamodule(new_config: DictConfig):
    datamodule = AnemoiMultipleDatasetsDataModule(new_config, None)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    train_loaders = datamodule.train_dataloader()

    assert isinstance(train_loaders, torch.utils.data.DataLoader)

    batch = next(iter(train_loaders))

    assert isinstance(batch, dict)
    assert batch["input"]["era5"].shape == (1, 2, 1, 40320, 12)
    assert batch["target"]["era5"].shape == (1, 1, 1, 40320, 9)


if __name__ == "__main__":
    pytest.main([__file__])
