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


def get_sample_config_dict(sample: DictConfig) -> Dict:
    return CONFIG
