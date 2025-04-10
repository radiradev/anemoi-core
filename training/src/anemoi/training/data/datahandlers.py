#
#
#
#
#
#


def str_(t):
    """Not needed, but useful for debugging"""
    import numpy as np

    if isinstance(t, np.ndarray):
        return t.shape
    if isinstance(t, (list, tuple)):
        return "[" + " , ".join(str_(e) for e in t) + "]"
    if isinstance(t, dict):
        return "{" + " , ".join(f"{k}: {str_(v)}" for k, v in t.items()) + "}"
    return str(t)


data_config = {
    "fields": {
        "dataset": "aifs-ea-an-oper-0001-mars-o48-2020-2021-6h-v1",
        "processors": {
            "normalizers": True,
            "imputers": True,
        },
    },
}
model_config = {
    "input": {
        "fields": ["q_500", "2t"]
        # "more": ["q_50", "2t"]
    },
    "output": {"fields": ["q_1000", "2t"]},
}

from anemoi.datasets.data import open_dataset


class DataHandlers(dict):
    def __init__(self, config):
        for name, kwargs in config.items():
            self[name] = create_data_handler(**kwargs)


class BaseDataHandler:
    def __init__(self, dataset, processors=None):
        self._dataset = open_dataset(dataset)
        self.processors = processors

    def __getitem__(self, i):
        return self._dataset[i]


class DataHandler(BaseDataHandler):
    pass


class SelectedDataHandler(BaseDataHandler):
    def __init__(self, dh, select=None):
        self._dataset = open_dataset(dh._dataset, select=select)
        self.processors = dh.processors


def create_data_handler(**kwargs):
    return DataHandler(**kwargs)


def select(data_handler, select=None):
    if select is None:
        return data_handler
    return SelectedDataHandler(data_handler, select=select)


class SampleProvider:
    def __init__(self, kwargs, datahandlers):

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


class Tester:
    def __init__(self, config, datahandlers):
        self.input = SampleProvider(config["input"], datahandlers)
        self.output = SampleProvider(config["output"], datahandlers)

    def getdata(self, i):
        input = self.input[i]
        output = self.output[i]
        print("input", str_(input))
        print("output", str_(output))


dhs = DataHandlers(data_config)
print(dhs)

model = Tester(model_config, dhs)
model.getdata(5)
