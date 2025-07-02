import pytest
import torch
from omegaconf import DictConfig

from anemoi.training.data.refactor.data_handlers import data_handler_factory
from anemoi.training.data.refactor.multiple_datasets_datamodule import AnemoiMultipleDatasetsDataModule
from anemoi.training.data.refactor.providers import sample_provider_factory


def _test_datahandler(new_config: DictConfig):
    config = new_config
    datahandler = data_handler_factory(config.data.data_handler, top_level=True)

    # TODO : test morethings here
    datahandler[0]
    datahandler.name_to_index
    datahandler.statistics
    datahandler.frequency
    datahandler.start_date
    datahandler.end_date
    datahandler.groups
    datahandler.variables
    # datahandler.latitudes # not implemented
    # datahandler.longitudes # not implemented


def _test_sampleprovider(new_config: DictConfig):
    config = new_config
    datahandler = data_handler_factory(config.data.data_handler, top_level=True)
    sample_provider = sample_provider_factory(
        **config.model.sample_structure,
        provider=datahandler,
    )
    sample_provider[0]


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
