import pytest
import torch
from omegaconf import DictConfig

from anemoi.training.data.refactor.multiple_datasets_datamodule import AnemoiMultipleDatasetsDataModule
from anemoi.training.data.refactor.draft import sample_provider_factory, Context
from anemoi.training.data.refactor.read_config import get_data_config_dict, get_sample_config_dict


def test_sampleprovider(new_config: DictConfig):
    dhs_config = get_data_config_dict(new_config.data.data_handlers)
    sample_config = get_sample_config_dict(new_config.model.sample)

    context = Context("training", data_config=dhs_config, start=2019, end=2020)

    sample_provider = sample_provider_factory(context=context, **sample_config)
    sample_provider[0]
    assert hasattr(sample_provider, "latitudes")
    assert hasattr(sample_provider, "longitudes")
    assert hasattr(sample_provider, "timedeltas")
    assert hasattr(sample_provider, "processors")

    assert set(sample_provider.processors(0).keys()) == {"input", "target"}


def test_datamodule(new_config: DictConfig):
    datamodule = AnemoiMultipleDatasetsDataModule(new_config, None)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    train_loaders = datamodule.train_dataloader()

    assert isinstance(train_loaders, torch.utils.data.DataLoader)

    batch = next(iter(train_loaders))

    assert isinstance(batch, dict)

    assert set(batch.keys()) == {"input", "target"}

    assert set(batch["input"].keys()) == {"era5", "amsr_h180"}
    assert set(batch["target"].keys()) == {"era5", "amsr_h180"}

    assert batch["input"]["era5"].shape == (1, 2, 7, 1, 40320) # (, time, vars, ens, latlon)
    assert len(batch["input"]["amsr_h180"]) == 1
    assert batch["input"]["amsr_h180"][0].shape == (1, 3, 187186)
    assert len(batch["target"]["era5"]) == 1
    assert len(batch["target"]["amsr_h180"]) == 1

if __name__ == "__main__":
    pytest.main([__file__])
