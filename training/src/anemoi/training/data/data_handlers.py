from anemoi.datasets.data import open_dataset
import logging
from dataclasses import dataclass
from enum import Enum

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


@dataclass
class RangeSplit:
    start: str = None
    end: str = None

    def to_dict(self) -> dict:
        return dict(start=self.start, end=self.end)


class Stage(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"

class DataSplits(dict):
    # TODO: Use timestamps??
    def __init__(self, config: dict[str, dict]):
        for stage in Stage:
            self[stage] = RangeSplit(start=config[stage.value].start, end=config[stage.value].end)
        
        self.check_training_end_specified()
        self.check_overlapping_stages()

    def check_overlapping_stages(self) -> None:
        if not self[Stage.TRAINING].end < self[Stage.VALIDATION].start:
            LOGGER.warning(
                "Training end date %s is not before validation start date %s.",
                self[Stage.TRAINING].end,
                self[Stage.VALIDATION].start,
            )

        assert self[Stage.TRAINING].end < self[Stage.TEST].start, (
            f"Training end date {self[Stage.TRAINING].end} is not before test start date {self[Stage.TEST].start}"
        )
        assert self[Stage.VALIDATION].end < self[Stage.TEST].start, (
            f"Validation end date {self[Stage.VALIDATION].end} is not before test start date {self[Stage.TEST].start}"
        )

    def check_training_end_specified(self) -> None:
        # Set the training end date if not specified
        if self[Stage.TRAINING].end is None:
            LOGGER.info(
                "No end date specified for training data, setting default before validation start date %s.",
                self[Stage.VALIDATION].start - 1,
            )
            self[Stage.TRAINING].end = self[Stage.VALIDATION].start - 1


class DataHandlers(dict):
    def __init__(self, config: dict[str, dict]):
        for name, kwargs in config.items():
            self[name] = DataHandler(**kwargs)

    def set_variables(self, model_sample: ModelSample) -> None:
        for name in self.keys():
            self[name].set_variables(
                input_variables=model_sample.input_variables(name),
                output_variables=model_sample.output_variables(name),
            )
    
class AnemoiDataReaders(dict):
    def __init__(self, data_handlers: DataHandlers, stage_range: RangeSplit):
        for name, data_handler in data_handlers.items():
            self[name] = data_handler.get_dataset(stage_range)


class BaseDataHandler:
    def __init__(self, dataset: str | dict, processors: list | None = None):
        self._dataset = dataset
        self.variables = None
        self.processors = processors

    def get_dataset(self, range: RangeSplit) -> "anemoi.datasets.data.dataset.Dataset":
        return open_dataset(self._dataset, select=self.variables, **range.to_dict())

    def set_variables(self, input_variables: list[str], output_variables: list[str]) -> None:
        self.variables = list(set(input_variables + output_variables))

class DataHandler(BaseDataHandler):
    pass

