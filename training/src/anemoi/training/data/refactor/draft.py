import json

import numpy as np
import yaml
from rich.console import Console
from rich.tree import Tree

from anemoi.datasets import open_dataset


class SampleProvider:
    def __init__(self, context):
        self.context = context

    def __repr__(self):
        console = Console(record=True, width=120)
        tree = self._build_tree()
        with console.capture() as capture:
            console.print(tree)
        return capture.get()

    def _build_tree(self, label="SampleProvider"):
        return Tree(label)

    def _check_item(self, item):
        if not isinstance(item, (int, np.integer)):
            raise TypeError(f"Not implemented for non-integer indexing {type(item)}")

    def shuffle(self, *args, **kwargs):
        return ShuffledSampleProvider(self, *args, **kwargs)

class ShuffledSampleProvider(SampleProvider):
    def __init__(self, sample, seed=None):
        super().__init__(sample.context)
        self.sample = sample
        self.seed = seed

    def __getitem__(self, item):
        self.sample._check_item(item)
        if self.seed is not None:
            np.random.seed(self.seed + item)
        return self.sample[item]

    def _build_tree(self, label="ShuffledSampleProvider"):
        tree = Tree(label)
        subtree = self.sample._build_tree(label=f"SampleProvider: {type(self.sample).__name__}")
        tree.add(subtree)
        return tree

class GroupedSampleProvider(SampleProvider):
    def __init__(self, context, dic):
        super().__init__(context)
        self._samples = {k: sample_factory(self.context, **v) for k, v in dic.items()}

    def __getitem__(self, item):
        self._check_item(item)
        if item == 0:
            item = item + 1  # ✅✅ TODO provide the correct lenght
        return {k: v[item] for k, v in self._samples.items()}

    def __len__(self):
        return 118  # ✅✅ TODO provide the correct lenght

    def _build_tree(self, label="GroupedSampleProvider"):
        tree = Tree(label)
        for k, v in self._samples.items():
            subtree = v._build_tree(label=f"{k}: {type(v).__name__}")
            tree.add(subtree)
        return tree


class StepSampleProvider(SampleProvider):
    def __init__(self, context, dic):
        super().__init__(context)
        self._samples = {k: sample_factory(context, **v) for k, v in dic.items()}

    def __getitem__(self, item):
        self._check_item(item)
        out = []
        for k, v in self._samples.items():
            if k == "_6h":
                item = item - 1
            elif k == "_0h":
                pass
            elif k == "p6h":
                item = item + 1
            else:
                raise ValueError(f"Unknown step {k} in StepSampleProvider")
            out.append(v[item])
        return out

    def _build_tree(self, label="GroupedSampleProvider"):
        tree = Tree(label)
        for k, v in self._samples.items():
            subtree = v._build_tree(label=f"{k}: {type(v).__name__}")
            tree.add(subtree)
        return tree


class Leaf(SampleProvider):
    def __init__(self, context, variables, group):
        super().__init__(context)
        self.group = group
        self.variables = variables

    def __getitem__(self, item):
        self._check_item(item)
        dh = DataHandler(self.context, self.group, item, variables=self.variables)
        dh = dh.load()
        return dh

    def _build_tree(self, label="Leaf"):
        return Tree(f"{label}  -> {self.group} variables={self.variables}")


class DataHandler:
    def __init__(self, context, group, args, variables=[]):
        self.context = context
        self.group = group
        self.args = args
        self.dataset = self.context.data_config[self.group]["dataset"]

        variables = [f"{group}.{v}" for v in variables]
        self.variables = variables

        self.config = dict(dataset=self.dataset, select=self.variables)
        self.ds = open_dataset(**self.config)
        print(f"🔍 Opened dataset with config: {self.config}")

    def load(self):
        return self.ds[self.args][self.group]

    def __repr__(self):
        return f"DataHandler({self.dataset} @ {self.group}, [{', '.join(self.variables)}], {self.args})"


class Context:
    def __init__(self, name="no-name", start=None, end=None, data_config=None):
        self.selection = dict(start=start, end=end)
        self.data_config = data_config
        self.name = name

    def __repr__(self):
        return f"Context(selection={self.selection})"


def sample_factory(context, **kwargs):
    kwargs = kwargs.copy()
    if isinstance(context, dict):
        context = Context(**context)
    if context is None:
        context = Context()
    if "GROUPS" in kwargs:
        return GroupedSampleProvider(context, kwargs["GROUPS"])
    if "STEPS" in kwargs:
        return StepSampleProvider(context, kwargs["STEPS"])
    if "variables" in kwargs:
        return Leaf(context, variables=kwargs["variables"], group=kwargs["data"])
    assert False, f"Unknown sample type for kwargs {kwargs}"


# TEST ---------------------------------
if __name__ == "__main__":
    CONFIG = dict(
        data=dict(
            #        era5=dict(
            # dataset=dict(dataset="aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8", set_group="era5"),
            # preprocessors=dict(
            #    tp=[dict(normalizer="mean-std")]),
            # ),
            #        ),
            snow=dict(dataset="observations-testing-2018-2018-6h-v0"),
            metop_a=dict(dataset="observations-testing-2018-2018-6h-v0"),
        ),
        training_selection=dict(
            # start=...,
            end="2018-11-01",
        ),
        validation_selection=dict(
            start="2018-11-02",
            # end=...,
        ),
        sample=dict(
            GROUPS=dict(
                input=dict(
                    GROUPS=dict(
                        #                    fields=dict(  # "fields" is a user defined key
                        #                        STEPS=dict(
                        #                            _6h=dict(
                        #                                variables=["q_50", "2t"],
                        #                                data="era5",
                        #                            ),
                        #                            _0h=dict(
                        #                                variables=["q_50", "2t"],
                        #                                data="era5",
                        #                            ),
                        #                        ),
                        #                    ),
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

    def show_yaml(structure):
        return yaml.dump(structure, indent=2, sort_keys=False)

    def show_json(structure):
        return json.dumps(shorten_numpy(structure), indent=2)

    def shorten_numpy(structure):
        if isinstance(structure, np.ndarray):
            return f"np.array({structure.shape})"
        if isinstance(structure, list):
            if all(isinstance(item, int) for item in structure):
                return "*" + "/".join(map(str, structure))
            return [shorten_numpy(item) for item in structure]
        if isinstance(structure, dict):
            return {k: shorten_numpy(v) for k, v in structure.items()}
        if isinstance(structure, DataHandler):
            return str(structure)
        return structure

    for k, v in CONFIG.items():
        print(f"💬 CONFIG {k}")
        print(show_yaml(v))
    print("-----------------")

    print("✅ SampleProvider")
    sample_config = CONFIG["sample"]
    training_context = Context(
        "training",
        data_config=CONFIG["data"],
        **CONFIG["training_selection"],
    )
    s = sample_factory(context=training_context, **sample_config)
    print(s)

    print("🆗 DataHandler")
    results_structure = s[3]

    print(show_json(results_structure))

    class Resolver:
        def __init__(self):
            self._mapping = {}
            self._count = 0

        def register(self, result):
            key = self._count
            self._mapping[key] = result
            self._count += 1
            return key

    def gather_results(resolver, structure):
        if isinstance(structure, DataHandler):
            return resolver.register(structure)
        if isinstance(structure, list):
            return [gather_results(resolver, item) for item in structure]
        if isinstance(structure, dict):
            return {k: gather_results(resolver, v) for k, v in structure.items()}
        raise TypeError(f"Unsupported type in results structure: {type(structure)}")

    # resolver = Resolver()
    # struct = gather_results(resolver, results_structure)
    # print("🔍 Gathered results structure:")
    # print(show_json(struct))
    # print("🔍 Resolver mapping:")
    # for k, v in resolver._mapping.items():
    #    print(f"{k}: {v}")
