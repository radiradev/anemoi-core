from omegaconf import DictConfig
import pytest
from anemoi.training.data.refactor.read_config import get_data_config_dict, get_sample_config_dict

from anemoi.training.data.refactor.read_config import CONFIG_OBS


def test_sample_config(new_config: DictConfig):
    sample_config = get_sample_config_dict(new_config.model.sample)
    assert sample_config == CONFIG_OBS


def test_data_config(new_config: DictConfig):
    dhs_config = get_data_config_dict(new_config.data.data_handlers)


if __name__ == "__main__":
    pytest.main([__file__])