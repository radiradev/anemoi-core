from typing import Dict
from omegaconf import DictConfig


CONFIG = dict(
    data=dict(
        era5=dict(
            dataset=dict(dataset="aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8", set_group="era5"),
            #preprocessors=dict(
            #    tp=[dict(normalizer="mean-std")]
            #),
        ),
        amsr_h180=dict(dataset="observations-testing-2018-2018-6h-v0", select=["amsr_h180.*"]),
    ),
    sample=dict(
        GROUPS=dict(
            input=dict(
                GROUPS=dict(
                    era5=dict(  # "fields" is a user defined key
                        STEPS={
                            "-6H": dict(
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
                            "-0H": dict(
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
                    amsr_h180=dict(  # "metar" is a user defined key
                        STEPS={
                            "0H": dict(
                                variables=["rawbt_1", "rawbt_2", "rawbt_3"],
                                data="amsr_h180",
                            ),
                        },
                    ),
                ),
            ),
            target=dict(
                GROUPS=dict(
                    era5=dict(  # "era5" is a user defined key
                        STEPS={
                            "6H": dict(
                                variables=["10u", "2t", "2d", "q_100", "q_1000"],
                                data="era5",
                            ),
                        },
                    ),
                    amsr_h180=dict(  # "amsr_h180" is a user defined key
                        STEPS={
                            "6H": dict(
                                variables=["rawbt_1", "rawbt_2", "rawbt_3"],
                                data="amsr_h180",
                            ),
                        },
                    ),
                ),
            ),
        ),
    ),
)


def get_config_dict(data: DictConfig, sample: DictConfig) -> Dict:
    return CONFIG
