from omegaconf import DictConfig
import pytest

@pytest.fixture
def new_config() -> DictConfig:
    return DictConfig(
    {
        "data": {
            "data_handlers": {
                "era5": {
                    "dataset": "aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6",
                    "processors": {
                        "normalizer": {
                            "_target_": "anemoi.models.preprocessing.normalizer.InputNormalizer",
                            "config": {
                                "default": "mean-std",
                                "std": ["tp"]
                            }
                        }
                    }
                }
            }
        },
        "model": {
            "model": {
                "input": {
                    "era5": [
                        "cos_latitude",
                        "cos_longitude",
                        "sin_latitude",
                        "sin_longitude",
                        "10u",
                        "10v",
                        "2t",
                        "2d",
                        "q_100",
                        "q_300",
                        "q_700",
                        "q_1000"
                    ],
                }, 
                "output": {
                    "era5": [
                        "10u",
                        "10v",
                        "2t",
                        "2d",
                        "q_100",
                        "q_300",
                        "q_700",
                        "q_1000",
                        "tp"
                    ]
                }
            }
        },
    }
    )