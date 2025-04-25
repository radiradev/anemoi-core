import pytest
import torch
from omegaconf import DictConfig

from anemoi.training.data.datamodule import AnemoiMultipleDatasetsDataModule


def test_datamodule(new_config: DictConfig):
    datamodule = AnemoiMultipleDatasetsDataModule(new_config, None)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    train_loaders = datamodule.train_dataloader()

    assert isinstance(train_loaders, dict)

    batch = {}
    for name, dl in train_loaders.items():
        assert isinstance(name, str)
        assert isinstance(dl, torch.utils.data.DataLoader)
        batch[name] = next(iter(dl))

    assert isinstance(batch, dict)


if __name__ == "__main__":
    pytest.main([__file__])
