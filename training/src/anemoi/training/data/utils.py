# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from omegaconf import OmegaConf, DictConfig
import pandas as pd
from datetime import datetime


class SamplerProviderName(str):
    pass


class DataHandlerName(str):
    pass


def parse_date(date: str | int) -> datetime:
    try:
        return pd.to_datetime(str(date))
    except ValueError as e:
        raise ValueError(f"Invalid date format: {e}")


def specify_datahandler_config(config: dict, key: str) -> dict:
    dataset = config[key]
    base_dataset = {"dataset": config["dataset"]} if isinstance(config["dataset"], str) else config["dataset"]

    if "dataset" not in dataset:
        dataset["dataset"] = base_dataset
    elif not isinstance(dataset["dataset"], str) and "dataset" not in dataset["dataset"]:
            dataset["dataset"] = OmegaConf.merge(base_dataset, dataset["dataset"])

    if "processors" not in dataset:
        dataset["processors"] = config["processors"]

    return dataset


def get_dataloader_config(config: DictConfig, field: str, keys_to_ignore: list = []) -> dict:
    cfg = {}
    for key, value in config.items():
        if key in keys_to_ignore:
            continue
        elif isinstance(value, str):
            cfg[key] = value
        elif isinstance(value, DictConfig) and key in value:
            cfg[key] = value[key]

    return cfg
