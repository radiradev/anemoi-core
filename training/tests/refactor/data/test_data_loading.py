from anemoi.training.data.datamodule import AnemoiMultipleDatasetsDataModule


def test_datamodule(new_config):
    datamodule = AnemoiMultipleDatasetsDataModule(new_config, None)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    train_loader = datamodule.train_dataloader()

    x, y = next(iter(train_loader))

    assert isinstance(x, dict)
    assert isinstance(y, dict)
