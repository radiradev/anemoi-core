from typing import Dict

from omegaconf import DictConfig

CONFIG_OBS = dict(
    dictionary=dict(
        input=dict(
            dictionary=dict(
                era5=dict(  # "fields" is a user defined key
                    tensor=[
                        dict(
                            timedelta="-6h",
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
                        dict(
                            timedelta="-0h",
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
                    ],
                ),
                amsr_h180=dict(  # "metar" is a user defined key
                    tuple=[
                        dict(
                            timedelta="0h",
                            variables=["rawbt_1", "rawbt_2", "rawbt_3"],
                            data="amsr_h180",
                        ),
                    ],
                ),
            ),
        ),
        target=dict(
            dictionary=dict(
                era5=dict(  # "era5" is a user defined key
                    tuple=[
                        dict(
                            timedelta="6h",
                            variables=["10u", "2t", "2d", "q_100", "q_1000"],
                            data="era5",
                        ),
                    ],
                ),
                amsr_h180=dict(  # "amsr2" is a user defined key
                    tuple=[
                        dict(
                            timedelta="6h",
                            variables=["rawbt_1", "rawbt_2", "rawbt_3"],
                            data="amsr_h180",
                        ),
                    ],
                ),
            ),
        ),
    ),
)

CONFIG_DOWNSCALING = dict(
    dictionary=dict(
        input=dict(
            dictionary=dict(
                era5=dict(
                    tensor=[
                        dict(
                            timedelta="0h",
                            variables=[
                                "cos_latitude",
                                "sin_latitude",
                                "cos_longitude",
                                "sin_longitude",
                                "10u",
                                "2t",
                                "2d",
                                "q_100",
                                "q_1000",
                            ],
                            data="era5",
                        )
                    ]
                ),
                cerra=dict(
                    tensor=[
                        dict(
                            timedelta="0h",
                            variables=[
                                "cos_latitude",
                                "sin_latitude",
                                "cos_longitude",
                                "sin_longitude",
                            ],
                            data="cerra",
                        )
                    ]
                ),
            ),
        ),
        target=dict(
            dictionary=dict(
                cerra=dict(
                    tensor=[
                        dict(
                            timedelta="0h",
                            variables=[
                                "tp",
                                "t_100"
                            ],
                            data="cerra",
                        )
                    ]
                    ),
                ),
        ),
    ),
)

DATA_CONFIG_OBS = dict(
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
    amsr_h180=dict(
        ## NOW: Only 1 group is supported for each "key" (dh)
        dataset=dict(
            dataset="/etc/ecmwf/nfs/dh1_home_a/mafp/work/obs/data/vz/obs-2018-11.vz",
            select=["amsr_h180.*"],
        ),
    ),
)


DATA_CONFIG_DOWNSCALING = dict(
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
    cerra=dict(
        ## NOW: Only 1 group is supported for each "key" (dh)
        dataset=dict(
            dataset="cerra-rr-an-oper-0001-mars-5p5km-1984-2020-6h-v2-hmsi",
            set_group="cerra",
        ),
    ),
)

def get_data_config_dict(data) -> Dict:
    return DATA_CONFIG_OBS

def get_sample_config_dict(sample: DictConfig) -> Dict:
    return CONFIG_OBS
