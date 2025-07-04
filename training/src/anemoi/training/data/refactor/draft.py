import json

import numpy as np
import yaml
from rich.console import Console
from rich.tree import Tree
from hydra.utils import instantiate

from anemoi.datasets import open_dataset
from anemoi.utils.dates import frequency_to_timedelta


class Context:
    def __init__(self, name="no-name", start=None, end=None, data_config=None):
        self.selection = dict(start=start, end=end)
        self.data_config = data_config
        self.name = name

    def __repr__(self):
        return f"Context(selection={self.selection})"


class SampleProvider:
    def __init__(self, context: Context):
        self.context = context

    def __getitem__(self, item):
        self._check_item(item)
        return self.get("__getitem__", item)

    def everything_as_a_dict(self, item):
        self._check_item(item)
        return self.get("everything-as-a-dict", item)

    def latitudes(self, item):
        self._check_item(item)
        return self.get("latitudes", item)

    def longitudes(self, item):
        self._check_item(item)
        return self.get("longitudes", item)

    def timedeltas(self, item):
        self._check_item(item)
        return self.get("timedeltas", item)

    def name_to_index(self, item):
        self._check_item(item)
        return self.get("name_to_index", item)
    
    def statistics(self, item):
        self._check_item(item)
        return self.get("statistics", item)

    def processors(self, item):
        processors = []
        for k, v in self.context.data_config.items():
            if not "processors" in v:
               continue

            for n, p in v["processors"].items():
                processors.append([
                    n, 
                    instantiate(
                        p, name_to_index_training_input=self.name_to_index(item)[k], statistics=self.statistics(item)[k]
                    )
                ])

        return processors

    @property
    def frequency(self):
        return frequency_to_timedelta("6h")

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
    def __init__(self, sample: SampleProvider, seed: int = None):
        super().__init__(sample.context)
        self.sample = sample
        self.seed = seed

    def get(self, what, item):
        self.sample._check_item(item)
        if self.seed is not None:
            np.random.seed(self.seed + item)
        return self.sample.get(what, item)

    def _build_tree(self, label="ShuffledSampleProvider"):
        tree = Tree(label)
        subtree = self.sample._build_tree(label=f"SampleProvider: {type(self.sample).__name__}")
        tree.add(subtree)
        return tree


class GroupedSampleProvider(SampleProvider):
    def __init__(self, context: Context, dictionary: dict, with_attributes: bool = False):
        super().__init__(context)
        self.with_attributes = with_attributes
        self._samples = {k: sample_factory(self.context, **v) for k, v in dictionary.items()}

    def __getattr__(self, key):
        if key in self._samples:
            return self._samples[key]
        raise AttributeError(f"{type(self).__name__} has no attribute '{key}'")

    def get(self, what, item):
        self._check_item(item)
        if item == 0:
            item = item + 1  # ✅✅ TODO provide the correct lenght
        return {k: v.get(what, item) for k, v in self._samples.items()}

    def __len__(self):
        return 118  # ✅✅ TODO provide the correct lenght

    def _build_tree(self, label="GroupedSampleProvider"):
        tree = Tree(label)
        for k, v in self._samples.items():
            subtree = v._build_tree(label=f"{k}: {type(v).__name__}")
            tree.add(subtree)
        return tree


class StepSampleProvider(SampleProvider):
    def __init__(self, context: Context, dic: dict):
        super().__init__(context)
        self._samples = {k: sample_factory(context, **v) for k, v in dic.items()}

    def get(self, what, item):
        self._check_item(item)
        out = []
        for k, v in self._samples.items():
            k = frequency_to_timedelta(k)
            sample_step = k / v.frequency
            assert sample_step == int(sample_step)
            out.append(v.get(what, item + int(sample_step)))
        return out

    def _build_tree(self, label="GroupedSampleProvider"):
        tree = Tree(label)
        for k, v in self._samples.items():
            subtree = v._build_tree(label=f"{k}: {type(v).__name__}")
            tree.add(subtree)
        return tree


class Leaf(SampleProvider):
    def __init__(self, context: Context, variables : list[str], group: str):
        super().__init__(context)
        self.group = group
        self.variables = variables

    def get(self, what, item):
        # this may be moved to the Mother class
        self._check_item(item)
        if isinstance(what, str):
            if what == "everything-as-a-dict":
                return self._get("__getitem__", "latitudes", "longitudes", "timedeltas", "name_to_index", item)
            else:
                out = self._get(what, item)
                assert len(out) == 1, f"Expected single item for {what}, got {len(out)}"
                key = list(out.keys())[0]
                return out[key]
        return self._get(*what, item=item)

    def _get(self, *what_and_item):
        *what, item = what_and_item
        dh = DataHandler(self.context, self.group, item, variables=self.variables)
        record = dh.record
        second = np.timedelta64(1, "s")

        data = {}
        for w in what:
            if w == "__getitem__":
                data["data"] = record[self.group]
            elif w == "latitudes":
                data["latitudes"] = record.latitudes[self.group]
            elif w == "longitudes":
                data["longitudes"] = record.longitudes[self.group]
            elif w == "timedeltas":
                data["timedeltas"] = record.timedeltas[self.group] // second
            elif w == "name_to_index":
                data["name_to_index"] = record.name_to_index[self.group]
            elif w == "statistics":
                data["statistics"] = record.statistics[self.group]
            else:
                raise ValueError(f"Unknown request '{w}' for Leaf sample provider")
        return data

    def _build_tree(self, label="Leaf"):
        return Tree(f"{label}  -> {self.group} variables={self.variables}")


class DataHandler:
    _record = None

    def __init__(self, context, group, args, variables=[]):
        self.context = context
        self.group = group
        self.args = args
        self.dataset = self.context.data_config[self.group]["dataset"]

        variables = [f"{group}.{v}" for v in variables]
        self.variables = variables

        self.config = dict(dataset=self.dataset, select=self.variables)
        self.ds = open_dataset(**self.config)
        # print(f"🔍 Opened dataset with config: {self.config}")
        self.frequency = frequency_to_timedelta(self.ds.frequency)


    @property
    def record(self):
        if self._record is not None:
            return self._record
        self._record = self.ds[self.args]
        return self._record

    def __repr__(self):
        return f"DataHandler({self.dataset} @ {self.group}, [{', '.join(self.variables)}], {self.args})"


def sample_factory(context, **kwargs):
    kwargs = kwargs.copy()
    if isinstance(context, dict):
        context = Context(**context)
    if context is None:
        context = Context()
    if "dictionary" in kwargs:
        return GroupedSampleProvider(context, **kwargs)
    if "GROUPS" in kwargs:
        return GroupedSampleProvider(context, dictionary=kwargs["GROUPS"])
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
            dictionary=dict(
                input=dict(
                    dictionary=dict(
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
                            STEPS={
                                "-6h": dict(
                                    # give_all_things=dict(
                                    variables=["scatss_1", "scatss_2"],
                                    data="metop_a",
                                    # )
                                ),
                            },
                        ),
                        snow=dict(  # "iasi" is a user defined key
                            STEPS={
                                "0h": dict(
                                    variables=["sdepth_0"],
                                    data="snow",
                                ),
                            },
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

    print("Latitudes and longitudes:")
    print(show_json(s.latitudes(3)))
    print(show_json(s.longitudes(3)))
    print("❌ Everything as a dict:")
    print(show_json(s.everything_as_a_dict(3)))

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
