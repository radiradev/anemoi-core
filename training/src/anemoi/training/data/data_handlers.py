import logging
from dataclasses import dataclass
from enum import Enum

from anemoi.datasets.data import open_dataset

LOGGER = logging.getLogger(__name__)


class ModelSample:
    def __init__(self, input_sample: dict[str, dict], output_sample: dict[str, dict]):
        self.input_sample = input_sample
        self.output_sample = output_sample

    def num_input_variables(self, key: str) -> int:
        return len(self.input_sample[key])

    def num_output_variables(self, key: str) -> int:
        return len(self.output_sample[key])

    def input_variables(self, key: str) -> list[str]:
        return self.input_sample[key]

    def output_variables(self, key: str) -> list[str]:
        return self.output_sample[key]


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


    def set_variables(self, model_sample: ModelSample) -> None:
        for name in self.keys():
            self[name].set_variables(
                input_variables=model_sample.input_variables(name),
                output_variables=model_sample.output_variables(name),
            )


class BaseDataHandler:
    def __init__(self, dataset: str | dict, processors: list | None = None):
        self._dataset = open_dataset(dataset)
        self.variables = None
        self.processors = processors


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
    def __init__(self, kwargs: dict, datahandlers: "DataHandlers"):

        self._variables = {}
        self._data_handlers = {}

        for key, variables in kwargs.items():
            if key not in datahandlers:
                raise ValueError(f"Unknown data handler: {key}, should be one of {list(datahandlers.keys())}")
            
            assert isinstance(variables, (list, tuple)), f"Select must be a list or tuple, not {type(variables)}"
            self._variables[key] = variables
            self._data_handlers[key] = select(datahandlers[key], variables)

    def __getitem__(self, i):
        sample = {}

        for k, dh in self._data_handlers.items():
            v = dh[i]

            actual = v.shape[0]
            expected = len(self._variables[k])
            assert actual == expected, f"Variable {k} has shape {actual} != {expected}, {v.shape}"
            sample[k] = v

        return sample