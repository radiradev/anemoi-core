import pytest
from omegaconf import DictConfig


@pytest.fixture
def new_config() -> DictConfig:
    return DictConfig(
        {
            "data": {
                "frequency": "6h",
                "timestep": "6h",
                "data_handler": {
                    "era5": {
                        "dataset": "aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6",
                        "processors": {
                            "normalizer": {
                                "_target_": "anemoi.models.preprocessing.normalizer.InputNormalizer",
                                "config": {"default": "mean-std", "std": ["tp"]},
                            },
                        },
                    },
                    "amsr_h180": {
                        ## NOW: Only 1 group is supported for each "key" (dh)
                        "dataset": {
                            "dataset": "/etc/ecmwf/nfs/dh1_home_a/mafp/work/obs/data/vz/obs-2018-11.vz",
                            "select": ["amsr_h180.*"],
                        },
                    },
                },
            },
            "dataloader": {
                "sampler": {
                    "training": {
                        "start": 2018,
                        "end": 2019,
                    },
                    "validation": {
                        "start": 2019,
                        "end": 2020,
                    },
                },
                "prefetch_factor": 2,
                "pin_memory": True,
                "persistent_workers": True,
                "grid_indices": {"_target_": "anemoi.training.data.grid_indices.FullGrid", "nodes_name": "data"},
                "read_group_size": 1,
                "num_workers": {
                    "training": 8,
                    "validation": 8,
                },
                "batch_size": {
                    "training": 2,
                    "validation": 4,
                },
                "limit_batches": {
                    "training": None,
                    "validation": None,
                },
            },
            "model": {
                "sample": {
                    "input": {
                        "groups": {
                            "era5": {
                                "variables": [
                                    "era5.cos_latitude",
                                    "era5.sin_latitude",
                                    "era5.10u",
                                    "era5.2t",
                                    "era5.2d",
                                    "era5.q_100",
                                    "era5.q_1000",
                                ],
                                "steps": ["-6h", "0h"],
                            },
                            "amsr2": {
                                "variables": ["amsr2.rawbt_1", "amsr2.rawbt_2", "amsr2.rawbt_3"],
                                "steps": ["0h"],
                            },
                        },
                    },
                    "target": {
                        "groups": {
                            "era5": {
                                "variables": ["era5.10u", "era5.10v", "era5.2t", "era5.q_1000", "era5.tp"],
                                "steps": ["6h"],
                            },
                            "amsr2": {
                                "variables": ["amsr2.rawbt_1", "amsr2.rawbt_2", "amsr2.rawbt_3"],
                                "steps": ["6h"],
                            },
                        },
                    },
                },
                "model": {"_target_": "anemoi.models.models.AnemoiMultiModel"},
                "encoder": {"_target_": "anemoi.models.layers.mapper.GraphTransformerForwardMapper"},
                "processor": {"_target_": "anemoi.models.layers.processor.GraphTransformerProcessor"},
                "decoder": {"_target_": "anemoi.models.layers.mapper.GraphTransformerBackwardMapper"},
            },
            "training": {
                "rollout": {
                    "start": 1,
                    "epoch_increment": 0,
                    "max": 1,
                },
            },
        },
    )
