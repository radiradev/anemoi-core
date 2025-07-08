from typing import Dict
from omegaconf import DictConfig, OmegaConf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_OBS = os.path.join(BASE_DIR, "configs", "config_obs.yaml")
CONFIG_DOWNSCALING = os.path.join(BASE_DIR, "configs", "config_downscaling.yaml")
CONFIG_MULTIPLE = os.path.join(BASE_DIR, "configs", "config_multiple.yaml")


def get_example(which: str) -> Dict:
    if which == "obs":
        return OmegaConf.to_container(OmegaConf.load(CONFIG_OBS), resolve=True)
    if which == "downscaling":
        return OmegaConf.to_container(OmegaConf.load(CONFIG_DOWNSCALING), resolve=True)
    if which == "multiple":
        return OmegaConf.to_container(OmegaConf.load(CONFIG_MULTIPLE), resolve=True)
    raise ValueError("Only supportings examples: obs, downscaling, multiple.")


def get_data_config_dict(data, which: str) -> Dict:
    return get_example(which)["data"]


def get_sample_config_dict(sample: DictConfig, which: str) -> Dict:
    return get_example(which)["sample"]
