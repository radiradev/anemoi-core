from anemoi.datasets.data import open_dataset

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


class BaseDataHandler:
    def __init__(self, dataset: str | dict, processors: list | None = None):
        self._dataset = dataset
        self.variables = None
        self.processors = processors

    def get_dataset(self, start: str = None, end: str = None):
        return open_dataset(self._dataset, select=self.variables, start=start, end=end)

    def set_variables(self, input_variables: list[str], output_variables: list[str]) -> None:
        self.variables = list(set(input_variables + output_variables))

class DataHandler(BaseDataHandler):
    pass

