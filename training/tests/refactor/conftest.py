import pytest
from omegaconf import DictConfig


@pytest.fixture
def new_config() -> DictConfig:
    return DictConfig(
        {
            "data": {
                "frequency": "6h",
                "timestep": "6h",
                "data_handlers": {
                    "era5": {
                        "dataset": "aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6",
                        "processors": {
                            "normalizer": {
                                "_target_": "anemoi.models.preprocessing.normalizer.InputNormalizer",
                                "config": {"default": "mean-std", "std": ["tp"]},
                            },
                        },
                    },
                },
            },
            "dataloader": {
                "training": {
                    "start": None,
                    "end": 2018,
                },
                "validation": {
                    "start": 2019,
                    "end": 2019,
                },
                "test": {
                    "start": 2020,
                    "end": None,
                },
                "prefetch_factor": 2,
                "pin_memory": True,
                "grid_indices": {
                    "_target_": "anemoi.training.data.grid_indices.FullGrid", 
                    "nodes_name": "data"
                },
                "read_group_size": 1,
                "num_workers": {
                    "training": 8,
                    "validation": 8,
                    "test": 8
                },
                "batch_size":{
                    "training": 2,
                    "validation": 4,
                    "test": 4
                },
                "limit_batches": {
                    "training": None,
                    "validation": None,
                    "test": 20
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
                            "q_1000",
                        ],
                    },
                    "output": {"era5": ["10u", "10v", "2t", "2d", "q_100", "q_300", "q_700", "q_1000", "tp"]},
                },
            },
            "training": {
                "multistep_input": 2,
                "rollout": {
                    "start": 1,
                    "epoch_increment": 0,
                    "max": 1,
                },
            },
        },
    )
