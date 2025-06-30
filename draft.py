import json

import numpy as np
import yaml
from rich.console import Console
from rich.tree import Tree

from anemoi.datasets import open_dataset

CONFIG = dict(
    data=dict(
        era5=dict(
            dataset=dict(dataset="aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8", set_group="era5"),
            # preprocessors=dict(
            #    tp=[dict(normalizer="mean-std")]),
            # ),
        ),
        snow=dict(dataset="observations-testing-2018-2018-6h-v0"),
        metop_a=dict(dataset="observations-testing-2018-2018-6h-v0"),
    ),
    sample=dict(
        GROUPS=dict(
            input=dict(
                GROUPS=dict(
                    fields=dict(  # "fields" is a user defined key
                        STEPS=dict(
                            _6h=dict(
                                variables=["q_50", "2t"],
                                data="era5",
                            ),
                            _0h=dict(
                                variables=["q_50", "2t"],
                                data="era5",
                            ),
                        ),
                    ),
                    # user-friendly config would be:
                    # fields=dict(
                    #     steps=['-6h', '0h'],
                    #     variables=["q_50", "2t"],
                    #     data="era5",
                    # ),
                    metop=dict(  # "metar" is a user defined key
                        STEPS=dict(
                            _6h=dict(
                                variables=["scatss_1", "scatss_2"],
                                data="metop_a",
                            ),
                        ),
                    ),
                    snow=dict(  # "iasi" is a user defined key
                        STEPS=dict(
                            _0h=dict(
                                variables=["sdepth_0"],
                                data="snow",
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
)


class Sample:
    def __init__(self, datahandlers):
        self.datahandlers = datahandlers

    def __repr__(self):
        console = Console(record=True, width=120)
        tree = self._build_tree()
        with console.capture() as capture:
            console.print(tree)
        return capture.get()

    def _build_tree(self, label="Sample"):
        return Tree(label)


class GroupedSample(Sample):
    def __init__(self, datahandlers, dic):
        super().__init__(datahandlers)
        self._samples = {k: sample_factory(**v) for k, v in dic.items()}

    def __getitem__(self, item):
        return {k: v[item] for k, v in self._samples.items()}

    def _build_tree(self, label="GroupedSample"):
        tree = Tree(label)
        for k, v in self._samples.items():
            subtree = v._build_tree(label=f"{k}: {type(v).__name__}")
            tree.add(subtree)
        return tree


class StepSample(Sample):
    def __init__(self, datahandlers, dic):
        super().__init__(datahandlers)
        self._samples = {k: sample_factory(**v) for k, v in dic.items()}

    def __getitem__(self, item):
        out = []
        for k, v in self._samples.items():
            if k == "_6h":
                out.append(v[item - 1])
            elif k == "_0h":
                out.append(v[item])
            elif k == "p6h":
                out.append(v[item + 1])
        return out

    def _build_tree(self, label="GroupedSample"):
        tree = Tree(label)
        for k, v in self._samples.items():
            subtree = v._build_tree(label=f"{k}: {type(v).__name__}")
            tree.add(subtree)
        return tree


class Leaf(Sample):
    def __init__(self, datahandlers, variables, data):
        super().__init__(datahandlers)
        self.data_key = data
        self.variables = variables

    def __getitem__(self, item):
        result = Result(self.data_key, item, variables=self.variables)
        return result.load()

    def _build_tree(self, label="Leaf"):
        return Tree(f"{label}  -> {self.data_key} variables={self.variables}")


def sample_factory(datahandlers=None, **kwargs):
    kwargs = kwargs.copy()
    if datahandlers is None:
        datahandlers = []
    if "GROUPS" in kwargs:
        return GroupedSample(datahandlers, kwargs["GROUPS"])
    if "STEPS" in kwargs:
        return StepSample(datahandlers, kwargs["STEPS"])
    if "variables" in kwargs:
        return Leaf(datahandlers, variables=kwargs["variables"], data=kwargs["data"])
    assert False, f"Unknown sample type for kwargs {kwargs}"


class Result:
    def __init__(self, datahandler_key, *args, variables=[], **kwargs):
        cfg = CONFIG["data"][datahandler_key]
        assert "select" not in cfg, (cfg, variables)
        variables = [f"{datahandler_key}.{v}" for v in variables]
        dh = DataHandler(datahandler_key, **cfg, select=variables)

        self.func = dh.__getitem__
        self.args = args
        self.kwargs = kwargs

    def load(self):
        return self.func(*self.args, **self.kwargs)

    def __repr__(self):
        inside = []
        inside += [str(arg) for arg in self.args]
        inside += [f"{k}={v}" for k, v in self.kwargs.items()]
        return f"Result({self.datahandler}  ({', '.join(inside)})"


class DataHandler:
    def __init__(self, name, **config):
        self.name = name
        if isinstance(config, str):
            config = dict(dataset=config)
        if isinstance(config["dataset"], str):
            config = dict(dataset=config)

        self.config = config
        self._config_str = " ".join(f"{k}={v}" for k, v in config.items())

    def is_grouped_dataset(self, ds):
        from anemoi.datasets.data.records import BaseRecordsDataset

        return isinstance(ds, BaseRecordsDataset)

    @property
    def ds(self):
        ds = open_dataset(**self.config["dataset"])
        print(f"🔍 Opened dataset {self.name} with config: {self._config_str}")
        if self.name not in ds.groups:
            raise ValueError(f"Group '{self.name}' not found in dataset. Available groups: {ds.groups}")
        ds = ds[self.name]
        print(f"   Available variables for group '{self.name}': {ds.variables}")
        return ds

    def __getitem__(self, item):
        data = self.ds[item]
        assert isinstance(data, np.ndarray), f"Expected np.array, got {type(data)}, {type(self.ds)}"
        return data
        return f"np.array ds[{item}] with ds from {self._config_str} "

    def __str__(self):
        return f"DataHandler({self._config_str})"


def show_yaml(structure):
    return yaml.dump(structure, indent=2, sort_keys=False)


def show_json(structure):
    return json.dumps(structure, indent=2, default=shorten_numpy)


def shorten_numpy(structure):
    if isinstance(structure, np.ndarray):
        return f"np.array({structure.shape})"
    return structure


for k, v in CONFIG.items():
    print(f"💬 CONFIG {k}")
    print(show_yaml(v))
print("-----------------")

print("✅ Sample")
s = sample_factory(**CONFIG["sample"])
print(s)

print("🆗 Result")
results_structure = s[3]

print(show_json(results_structure))

# print("--------✅✅")
# internal = s._samples["fields"]
# result = internal[54, 0:1000]
# print(show_json(result))
