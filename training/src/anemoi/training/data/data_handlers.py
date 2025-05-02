import logging
from enum import Enum
import torch
import einops
from anemoi.datasets.data import open_dataset
from torch.utils.data import IterableDataset

LOGGER = logging.getLogger(__name__)


class Stage(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"


class DataHandlers(dict):
    def __init__(self, config: dict[str, dict]):
        for name, kwargs in config.items():
            self[name] = DataHandler(**kwargs)

    def check_no_overlap(self, other: "DataHandlers") -> None:
        for key, dh in self.items():
            if other[key].start_date < dh.end_date:
                raise ValueError()
            
            # TODO: What do we want to check ???
            # no_overlap vs is_completely_before 


class BaseDataHandler:
    def __init__(self, dataset: str | dict, processors: list | None = None):
        self._dataset = open_dataset(dataset)
        self.variables = None
        self.processors = processors

    @property
    def start_date(self): ...

    @property
    def end_date(self): ...


class DataHandler(BaseDataHandler):
    pass


class SelectedDataHandler(BaseDataHandler):
    def __init__(self, dh, select=None):
        self._dataset = open_dataset(dh._dataset, select=select)
        self.processors = dh.processors


def select(data_handler, select=None):
    if select is None:
        return data_handler
    return SelectedDataHandler(data_handler, select=select)


class SampleProvider:
    def __init__(self, kwargs: dict, datahandlers: "DataHandlers") -> None:
        self._variables = {}
        self._data_handlers = {}
        self._steps = {}

        for key, provider_config in kwargs.items():
            if key not in datahandlers:
                raise ValueError(f"Unknown data handler: {key}, should be one of {list(datahandlers.keys())}")
            
            variables = list(provider_config["variables"])
            assert isinstance(variables, (list, tuple)), f"variables must be a list or tuple, not {type(variables)}"
            self._variables[key] = variables
            self._data_handlers[key] = select(datahandlers[key], variables)
            self._steps[key] = provider_config["steps"]

    @property
    def keys(self) -> list[str]:
        return list(self._data_handlers.keys())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source, i = index
        time_indices = list(i + s for s in self._steps[source])
        x = self._data_handlers[source]._dataset[time_indices, :, :, :]
        x = einops.rearrange(x, "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")
        self.ensemble_dim = 1
        return torch.from_numpy(x)
    

class NativeGridMultDataset(IterableDataset):
    """Iterable dataset for AnemoI data on the arbitrary grids."""

    def __init__(
        self,
        data_reader: SampleProvider,
        shuffle: bool = True,
    ) -> None:
        self.data = data_reader
        self.shuffle = shuffle

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

    def __iter__(self) -> dict[str, torch.Tensor]:
        """Return an iterator over the dataset.

        The datasets are retrieved by anemoi.datasets from anemoi datasets. This iterator yields
        chunked batches for DDP and sharded training.

        Currently it receives data with an ensemble dimension, which is discarded for
        now. (Until the code is "ensemble native".)
        """
        for i in list(range(20, 30)):
            x = {}
            for key in self.data.keys:
                x[key] = self.data[key, i]
            yield x
