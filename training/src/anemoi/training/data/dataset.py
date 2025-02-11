# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
import os
import random
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Callable

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info

from anemoi.training.utils.seeding import get_base_seed
from anemoi.training.utils.usable_indices import get_usable_indices

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from anemoi.training.data.grid_indices import BaseGridIndices


class NativeGridDataset(IterableDataset):
    """Iterable dataset for AnemoI data on the arbitrary grids."""

    def __init__(
        self,
        data_reader: Callable,
        grid_indices: type[BaseGridIndices],
        rollout: int = 1,
        multistep: int = 1,
        timeincrement: int = 1,
        shuffle: bool = True,
        label: str = "generic",
        effective_bs: int = 1,
    ) -> None:
        """Initialize (part of) the dataset state.

        Parameters
        ----------
        data_reader : Callable
            user function that opens and returns the zarr array data
        grid_indices : Type[BaseGridIndices]
            indices of the grid to keep. Defaults to None, which keeps all spatial indices.
        rollout : int, optional
            length of rollout window, by default 12
        timeincrement : int, optional
            time increment between samples, by default 1
        multistep : int, optional
            collate (t-1, ... t - multistep) into the input state vector, by default 1
        shuffle : bool, optional
            Shuffle batches, by default True
        label : str, optional
            label for the dataset, by default "generic"
        effective_bs : int, default 1
            effective batch size useful to compute the lenght of the dataset
        """
        self.label = label
        self.effective_bs = effective_bs

        self.data = data_reader

        self.rollout = rollout
        self.timeincrement = timeincrement
        self.grid_indices = grid_indices

        # lazy init
        self.n_samples_per_epoch_total: int = 0
        self.n_samples_per_epoch_per_worker: int = 0

        # lazy init model and reader group info, will be set by the DDPGroupStrategy:
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1
        self.model_comm_group_id = 0
        self.global_rank = 0

        self.reader_group_rank = 0
        self.reader_group_size = 1

        self.seed_comm_num_groups = 1
        self.seed_comm_group_id = 0

        # additional state vars (lazy init)
        self.n_samples_per_worker = 0
        self.chunk_index_range: np.ndarray | None = None
        self.shuffle = shuffle

        # Data dimensions
        self.multi_step = multistep
        assert self.multi_step > 0, "Multistep value must be greater than zero."
        self.ensemble_dim: int = 2
        self.ensemble_size = self.data.shape[self.ensemble_dim]

    @cached_property
    def statistics(self) -> dict:
        """Return dataset statistics."""
        return self.data.statistics

    @cached_property
    def metadata(self) -> dict:
        """Return dataset metadata."""
        return self.data.metadata()

    @cached_property
    def supporting_arrays(self) -> dict:
        """Return dataset supporting_arrays."""
        return self.data.supporting_arrays()

    @cached_property
    def name_to_index(self) -> dict:
        """Return dataset statistics."""
        return self.data.name_to_index

    @cached_property
    def resolution(self) -> dict:
        """Return dataset resolution."""
        return self.data.resolution

    @cached_property
    def valid_date_indices(self) -> np.ndarray:
        """Return valid date indices.

        A date t is valid if we can sample the sequence
            (t - multistep + 1, ..., t + rollout)
        without missing data (if time_increment is 1).

        If there are no missing dates, total number of valid ICs is
        dataset length minus rollout minus additional multistep inputs
        (if time_increment is 1).
        """
        return get_usable_indices(self.data.missing, len(self.data), self.rollout, self.multi_step, self.timeincrement)

    def set_comm_group_info(
        self,
        global_rank: int,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        """Set model and reader communication group information (called by DDPGroupStrategy).

        Parameters
        ----------
        global_rank : int
            Global rank
        model_comm_group_id : int
            Model communication group ID
        model_comm_group_rank : int
            Model communication group rank
        model_comm_num_groups : int
            Number of model communication groups
        reader_group_rank : int
            Reader group rank
        reader_group_size : int
            Reader group size
        """
        self.global_rank = global_rank
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

        self.seed_comm_group_id = model_comm_group_id
        self.seed_comm_num_groups = model_comm_num_groups
        
        assert self.reader_group_size >= 1, "reader_group_size must be positive"

        LOGGER.debug(
            "NativeGridDataset.set_group_info(): global_rank %d, model_comm_group_id %d, "
            "model_comm_group_rank %d, model_comm_num_groups %d, reader_group_rank %d",
            global_rank,
            model_comm_group_id,
            model_comm_group_rank,
            model_comm_num_groups,
            reader_group_rank,
        )

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

        # Divide this equally across shards (one shard per group!)
        shard_size = len(self.valid_date_indices) // self.seed_comm_num_groups
        shard_start = self.seed_comm_group_id * shard_size
        shard_end = (self.seed_comm_group_id + 1) * shard_size

        shard_len = shard_end - shard_start
        self.n_samples_per_worker = shard_len // n_workers

        low = shard_start + worker_id * self.n_samples_per_worker
        high = min(shard_start + (worker_id + 1) * self.n_samples_per_worker, shard_end)
        self.chunk_index_range = np.arange(low, high, dtype=np.uint32)

        LOGGER.info(
            "Worker %d (pid %d, global_rank %d, model comm group %d) "
            " has low/high range %d / %d",
            worker_id,
            os.getpid(),
            self.global_rank,
            self.model_comm_group_id,
            low,
            high,
        )

        base_seed = get_base_seed()

        torch.manual_seed(base_seed)
        random.seed(base_seed)
        self.rng = np.random.default_rng(seed=base_seed)
        sanity_rnd = self.rng.random(1)

        LOGGER.info(
            (
                "Worker %d (%s, pid %d, glob. rank %d, model comm group %d, "
                "group_rank %d, seed group id %d, base_seed %d, sanity rnd %f)"
            ),
            worker_id,
            self.label,
            os.getpid(),
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            self.seed_comm_group_id,
            base_seed,
            sanity_rnd,
        )

    def __iter__(self) -> torch.Tensor:
        """Return an iterator over the dataset.

        The datasets are retrieved by Anemoi Datasets from zarr files. This iterator yields
        chunked batches for DDP and sharded training.

        Currently it receives data with an ensemble dimension, which is discarded for
        now. (Until the code is "ensemble native".)
        """
        if self.shuffle:
            shuffled_chunk_indices = self.rng.choice(
                self.valid_date_indices,
                size=len(self.valid_date_indices),
                replace=False,
            )[self.chunk_index_range]
        else:
            shuffled_chunk_indices = self.valid_date_indices[self.chunk_index_range]

        LOGGER.debug(
            (
                "Worker pid %d, label %s, worker id %d, global_rank %d, "
                "model comm group %d, group_rank %d, seed comm group id %d, using indices[0:10]: %s"
            ),
            os.getpid(),
            self.label,
            self.worker_id,
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            self.seed_comm_group_id,
            shuffled_chunk_indices[:10],
        )

        for i in shuffled_chunk_indices:
            start = i - (self.multi_step - 1) * self.timeincrement
            end = i + (self.rollout + 1) * self.timeincrement

            grid_shard_indices = self.grid_indices.get_shard_indices(self.reader_group_rank)
            if isinstance(grid_shard_indices, slice):
                # Load only shards into CPU memory
                x = self.data[start : end : self.timeincrement, :, :, grid_shard_indices]
            else:
                # Load full grid in CPU memory, select grid_shard after
                # Note that anemoi-datasets currently doesn't support slicing + indexing
                # in the same operation.
                x = self.data[start : end : self.timeincrement, :, :, :]
                x = x[..., grid_shard_indices]  # select the grid shard
            x = rearrange(x, "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")
            self.ensemble_dim = 1

            yield torch.from_numpy(x)

    def __repr__(self) -> str:
        return f"""
            {super().__repr__()}
            Dataset: {self.data}
            Rollout: {self.rollout}
            Multistep: {self.multi_step}
            Timeincrement: {self.timeincrement}
        """


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
        LOGGER.error("worker_info is None! Set num_workers > 0 in your dataloader!")
        raise RuntimeError
    dataset_obj = worker_info.dataset  # the copy of the dataset held by this worker process.
    dataset_obj.per_worker_init(
        n_workers=worker_info.num_workers,
        worker_id=worker_id,
    )


class EnsNativeGridDataset(NativeGridDataset):
    """Iterable ensemble dataset for AnemoI data on the arbitrary grids."""

    def __init__(
        self,
        data_reader: Callable,
        grid_indices: type[BaseGridIndices],
        rollout: int = 1,
        multistep: int = 1,
        timeincrement: int = 1,
        shuffle: bool = True,
        label: str = "generic",
        effective_bs: int = 1,
        ens_members_per_device: int = 1,
        num_gpus_per_ens: int = 1,
        num_gpus_per_model: int = 1,
    ) -> None:
        """Initialize (part of) the dataset state.

        Parameters
        ----------
        data_reader : Callable
            user function that opens and returns the zarr array data
        rollout : int, optional
            length of rollout window, by default 12
        multistep : int, optional
            collate (t-1, ... t - multistep) into the input state vector, by default 1
        ens_members_per_device: int, optional
            number of ensemble members input for each GPU device, by default 1
        shuffle : bool, optional
            Shuffle batches, by default True

        Raises
        ------
        RuntimeError
            Multistep value cannot be negative.
        """
        super().__init__(
            data_reader=data_reader,
            rollout=rollout,
            multistep=multistep,
            timeincrement=timeincrement,
            shuffle=shuffle,
            grid_indices=grid_indices,
            label=label,
            effective_bs=effective_bs,
        )

        # Lazy init
        self.ens_members_per_device = ens_members_per_device
        self.num_gpus_per_ens = num_gpus_per_ens
        self.num_gpus_per_model = num_gpus_per_model

        # lazy init model and reader group info, will be set by the DDPGroupStrategy:
        self.ens_comm_group_rank = 0
        self.ens_comm_num_groups = 1
        self.ens_comm_group_id = 0

    @property
    def num_eda_members(self) -> int:
        """Return number of EDA members."""
        return self.data.shape[2] - 1

    @property
    def eda_flag(self) -> bool:
        """Return whether EDA is enabled."""
        return self.data.shape[2] > 1

    def sample_eda_members(self, num_eda_members: int = 9) -> np.ndarray:
        """Subselect EDA ensemble members assigned to the current device."""
        tot_ens = self.ens_members_per_device * self.num_gpus_per_ens // self.num_gpus_per_model

        assert tot_ens <= num_eda_members, f"Can't generate an ensemble of size {tot_ens} from {num_eda_members} EDA perturbations"

        eda_member_gen_idx = self.rng.choice(range(num_eda_members), size=tot_ens, replace=False)
        offset = 1  # index=0 analysis, index=1 EDA recentred
        eda_member_gen_idx += offset

        effective_rank = self.ens_comm_group_rank // self.num_gpus_per_model
        eda_member_idx = np.sort(
            eda_member_gen_idx[effective_rank * self.ens_members_per_device : self.ens_members_per_device * (1 + effective_rank)],
        )

        LOGGER.debug(
            "GPU with global rank %s, comm_group_id %s, comm_group_rank %s will receive EDA member(s) %s",
            self.global_rank,
            self.ens_comm_group_id,
            self.ens_comm_group_rank,
            eda_member_gen_idx,
        )

        return eda_member_gen_idx, eda_member_idx

    def set_comm_group_info(
        self,
        global_rank: int,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        ens_comm_group_id: int,
        ens_comm_group_rank: int,
        ens_comm_num_groups: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        """Set model and reader communication group information (called by DDPGroupStrategy).

        Parameters
        ----------
        global_rank : int
            Global rank
        model_comm_group_id : int
            Model communication group ID
        model_comm_group_rank : int
            Model communication group rank
        model_comm_num_groups : int
            Number of model communication groups
        ens_comm_group_id : int
            Ensemble communication group ID
        ens_comm_group_rank : int
            Ensemble communication group rank
        ens_comm_num_groups : int
            Number of ensemble communication groups
        reader_group_rank : int
            Reader group rank
        reader_group_size : int
            Reader group size
        """
        self.global_rank = global_rank
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.ens_comm_group_id = ens_comm_group_id
        self.ens_comm_group_rank = ens_comm_group_rank
        self.ens_comm_num_groups = ens_comm_num_groups
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

        self.seed_comm_group_id = ens_comm_group_id  # used for seeding
        self.seed_comm_num_groups = ens_comm_num_groups

        assert self.reader_group_size >= 1, "reader_group_size must be positive"

        LOGGER.info(
            "NativeGridDataset.set_group_info(): global_rank %d, model_comm_group_id %d, "
            "model_comm_group_rank %d, model_comm_num_groups %d, reader_group_rank %d",
            global_rank,
            model_comm_group_id,
            model_comm_group_rank,
            model_comm_num_groups,
            reader_group_rank,
        )

        LOGGER.info(
            "NativeGridDataset.set_group_info(): global_rank %d, ens_comm_group_id %d, "
            "ens_comm_group_rank %d, ens_comm_num_groups %d, reader_group_rank %d",
            global_rank,
            ens_comm_group_id,
            ens_comm_group_rank,
            ens_comm_num_groups,
            reader_group_rank,
        )

    def __iter__(self):
        """Return an iterator over the dataset.

        The datasets are retrieved by Anemoi Datasets from zarr files. This iterator yields
        chunked batches for DDP and sharded training.
        """
        if self.shuffle:
            shuffled_chunk_indices = self.rng.choice(
                self.valid_date_indices,
                size=len(self.valid_date_indices),
                replace=False,
            )[self.chunk_index_range]
        else:
            shuffled_chunk_indices = self.valid_date_indices[self.chunk_index_range]

        LOGGER.debug(
            (
                "Worker pid %d, label %s, worker id %d, global_rank %d, "
                "model comm group %d, group_rank %d, seed comm group id %d, using indices[0:10]: %s"
            ),
            os.getpid(),
            self.label,
            self.worker_id,
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            self.seed_comm_group_id,
            shuffled_chunk_indices[:10],
        )

        for i in shuffled_chunk_indices:
            # start and end time indices, for analysis and EDA
            start = i - (self.multi_step - 1) * self.timeincrement
            end_an = i + (self.rollout + 1) * self.timeincrement
            end_eda = i + self.timeincrement

            if self.eda_flag:
                assert 1 > 2, "random stream is needed here for eda member sampling"
                eda_member_gen_idx, eda_member_idx = self.sample_eda_members(self.num_eda_members)
            else:
                eda_member_gen_idx = None
                eda_member_idx = None

            grid_shard_indices = self.grid_indices.get_shard_indices(self.reader_group_rank)
            if isinstance(grid_shard_indices, slice):
                # Load only shards into CPU memory
                x_an = self.data[start : end_an : self.timeincrement, :, 0:1, grid_shard_indices]
            else:
                # Load full grid in CPU memory, select grid_shard after
                # Note that anemoi-datasets currently doesn't support slicing + indexing
                # in the same operation.
                x_an = self.data[start : end_an : self.timeincrement, :, 0:1, ...]
                x_an = x_an[..., grid_shard_indices]  # select the grid shard
            x_an = rearrange(x_an, "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")

            x_pert: Optional[torch.Tensor] = None
            if self.eda_flag:
                if isinstance(grid_shard_indices, slice):
                    x_pert = self.data[start : end_eda : self.timeincrement, ..., grid_shard_indices]
                else:
                    x_pert = self.data[start : end_eda : self.timeincrement, ...]
                    x_pert = x_pert[..., grid_shard_indices]  # select the grid shard
                x_pert = rearrange(
                        x_pert[:, :, eda_member_idx, ...],
                        "dates variables ensemble gridpoints -> dates ensemble gridpoints variables",
                    )

            if x_pert is not None:
                sample = (torch.from_numpy(x_an), torch.from_numpy(x_pert),)
            else:
                sample = (torch.from_numpy(x_an),)

            yield sample

    def __repr__(self) -> str:
        return (
            f"""
            {super().__repr__()}
            Dataset: {self.data}
            Rollout: {self.rollout}
            Multistep: {self.multi_step}
            Timeincrement: {self.timeincrement}
            EDA: {self.eda_flag}
            """
            f"""
            Number of EDA members:
            {self.num_eda_members}" if self.eda_flag else
            """
        )
