from typing import Dict

from omegaconf import DictConfig

CONFIG = dict(
    GROUPS=dict(
        input=dict(
            GROUPS=dict(
                era5=dict(  # "fields" is a user defined key
                    STEPS={
                        "-6h": dict(
                            variables=[
                                "cos_latitude",
                                "sin_latitude",
                                "10u",
                                "2t",
                                "2d",
                                "q_100",
                                "q_1000",
                            ],
                            data="era5",
                        ),
                        "-0h": dict(
                            variables=[
                                "cos_latitude",
                                "sin_latitude",
                                "10u",
                                "2t",
                                "2d",
                                "q_100",
                                "q_1000",
                            ],
                            data="era5",
                        ),
                    },
                ),
                amsr2=dict(  # "metar" is a user defined key
                    STEPS={
                        "0h": dict(
                            variables=["rawbt_1", "rawbt_2", "rawbt_3"],
                            data="amsr2",
                        ),
                    },
                ),
            ),
        ),
        target=dict(
            GROUPS=dict(
                era5=dict(  # "era5" is a user defined key
                    STEPS={
                        "6h": dict(
                            variables=["10u", "2t", "2d", "q_100", "q_1000"],
                            data="era5",
                        ),
                    },
                ),
                amsr2=dict(  # "amsr2" is a user defined key
                    STEPS={
                        "6H": dict(
                            variables=["rawbt_1", "rawbt_2", "rawbt_3"],
                            data="amsr2",
                        ),
                    },
                ),
            ),
        ),
    ),
)


DATA_CONFIG = dict(
    era5=dict(
        dataset=dict(
            dataset="aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6",
            set_group="era5" 
        ),
        processors=dict(
            normalizer=dict(
                _target_="anemoi.models.preprocessing.normalizer.InputNormalizer",
                config=dict(default="mean-std", std=["tp"]),
            ),
        ),
    ),
    amsr2=dict(
        ## NOW: Only 1 group is supported for each "key" (dh)
        dataset=dict(
            dataset="/etc/ecmwf/nfs/dh1_home_a/mafp/work/obs/data/vz/obs-2018-11.vz",
            select=["amsr_h180.*"],
        ),
    ),
)

def get_data_config_dict(data) -> Dict:
    return DATA_CONFIG


def get_sample_config_dict(sample: DictConfig) -> Dict:
    return CONFIG
