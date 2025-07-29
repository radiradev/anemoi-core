import json
import os
import random
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info

CONFIG_YAML = """
sources:
  training:
    # era5:
    #   dataset:
    #     dataset: aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8
    #     set_group: era5
    #   # preprocessors:
    #   #   tp:
    #   #     - normalizer: mean-std
    snow:
        dataset: observations-testing-2018-2018-6h-v1-one-month
    metop_a:
        dataset: observations-testing-2018-2018-6h-v1-one-month
    amsr2_h180:
        dataset: observations-testing-2018-2018-6h-v1-one-month
  validation:
    todo:
sample:
    dictionary:
        input:
            dictionary:
                ascat_metop_a:
                    tuple:
                      - timedelta: "-6h"
                        variables:
                           metop_a: ["scatss_1", "scatss_2"]
                snow:
                    tuple:
                      - timedelta: "0h"
                        variables:
                           snow: ["sdepth_0"]
                amsr2:
                    tuple:
                      - timedelta: "-6h"
                        variables:
                            amsr2_h180: ["rawbt_1", "rawbt_2", "rawbt_3", "rawbt_4"]
"""

CONFIG = yaml.safe_load(CONFIG_YAML)


from anemoi.training.data.refactor.draft import sample_provider_factory


def show_yaml(structure):
    return yaml.dump(structure, indent=2, sort_keys=False)


def show_json(structure):
    return json.dumps(structure, indent=2, default=shorten_numpy)


def shorten_numpy(structure):
    if isinstance(structure, np.ndarray):
        return f"np.array({structure.shape})"
    return structure


def get_base_seed():
    """Get a base seed for random number generation.
    This is a placeholder function; replace with actual logic to get a base seed.
    """
    return 42  # Example fixed seed, replace with actual logic as needed


class DOPDataset(IterableDataset):
    def __init__(
        self,
        # config: dict,
        shuffle: bool = True,
        rollout: int = 1,
        multistep: int = 1,
        task: str = "training",
    ) -> None:

        self.shuffle = shuffle
        # self.config = config
        self.rollout = rollout
        self.multistep = multistep
        self.task = task

        # lazy init
        self.n_samples_per_epoch_total: int = 0
        self.n_samples_per_epoch_per_worker: int = 0

        # additional state vars (lazy init)
        self.n_samples_per_worker = 0
        self.chunk_index_range: Optional[np.ndarray] = None
        self.shuffle = shuffle
        self.rng: Optional[np.random.Generator] = None
        self.worker_id: int = -1

        # "full" shuffling
        self.data_indices: Optional[np.ndarray] = None

        self.seed_comm_group_id = 0
        self.seed_comm_num_groups = 1

        training_context = {
            "name": "training",
            "sources": CONFIG["sources"]["training"],
            "start": "2018-11-02",
            "end": "2018-11-01",
        }

        self._sample_provider = sample_provider_factory(context=training_context, **CONFIG["sample"])
        self._sample_provider = self._sample_provider.shuffle(seed=42)

        # self.len = len(self._sample_provider)

    def __get_sample(self, index: int):
        """Get a sample from the dataset."""
        return self._sample_provider[index]

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Called by worker_init_func on each copy of dataset.

        This initialises after the worker process has been spawned.

        Parameters
        ----------
        n_workers : int
            Number of workers
        worker_id : int
            Worker ID
        """
        self.worker_id = worker_id

        lenght = len(self._sample_provider)
        # Divide this equally across shards (one shard per group!)
        shard_size = lenght // self.seed_comm_num_groups
        shard_start = self.seed_comm_group_id * shard_size
        shard_end = min((self.seed_comm_group_id + 1) * shard_size, lenght)

        shard_len = shard_end - shard_start
        self.n_samples_per_worker = shard_len // n_workers

        low = shard_start + worker_id * self.n_samples_per_worker
        high = min(shard_start + (worker_id + 1) * self.n_samples_per_worker, shard_end)
        self.chunk_index_range = np.arange(low, high, dtype=np.uint32)

        seed = get_base_seed()  # all workers get the same seed (so they all get the same index shuffle)
        torch.manual_seed(seed)
        random.seed(seed)
        self.rng = np.random.default_rng(seed=seed)
        sanity_rnd = self.rng.random(1)
        print("Sanity check random number:", sanity_rnd)

    def __iter__(self):
        # no shuffle, just iterate over the chunk indices
        for idx in self.chunk_index_range:
            print(
                f"VALIDATION: Worker {self.worker_id} (pid {os.getpid()}) fetching sample index {idx} ...",
            )
            yield self.__get_sample(idx)


def worker_init_func(worker_id: int) -> None:
    """Configures each dataset worker process.

    Calls WeatherBenchDataset.per_worker_init() on each dataset object.

    Parameters
    ----------
    worker_id : int
        Worker ID

    Raises
    ------
    RuntimeError
        If worker_info is None
    """
    worker_info = get_worker_info()  # information specific to each worker process
    if worker_info is None:
        print("worker_info is None! Set num_workers > 0 in your dataloader!")
        raise RuntimeError
    dataset_obj = worker_info.dataset  # the copy of the dataset held by this worker process.
    dataset_obj.per_worker_init(
        n_workers=worker_info.num_workers,
        worker_id=worker_id,
    )


if __name__ == "__main__":

    ds = DOPDataset(
        # CONFIG,
        shuffle=False,
        rollout=1,
        multistep=1,
        task="training",
    )

    loader_params = {
        "batch_size": 1,  # must be 1 for the time being
        "batch_sampler": None,
        "num_workers": 2,
        "pin_memory": False,
        "worker_init_fn": worker_init_func,
        # "collate_fn": None, # collator_wrapper(return_original_metadata=cfg_.dataloader.return_dates),
    }

    dl = torch.utils.data.DataLoader(ds, **loader_params, sampler=None)

    for batch_idx, batch in enumerate(dl):
        print("%s", batch)
        if batch_idx >= 1:
            break
