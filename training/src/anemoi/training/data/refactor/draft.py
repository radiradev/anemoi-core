import json

import numpy as np
import yaml
from hydra.utils import instantiate
from rich.console import Console
from rich.tree import Tree

from anemoi.datasets import open_dataset
from anemoi.utils.dates import frequency_to_string
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

    def __len__(self):
        return len(self.range)

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
        self._check_item(item)
        return self.get("processors", item)

    @property
    def frequency(self):
        return frequency_to_timedelta("6h")

    def __repr__(self):
        console = Console(record=True, width=120)
        tree = self._build_tree()
        with console.capture() as capture:
            console.print(tree)
        return capture.get()

    def _build_tree(self, label=None):
        raise NotImplementedError("Subclasses must implement _build_tree method")

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
        length = len(self.sample)
        self.idx = np.arange(length)
        if seed is not None:
            np.random.seed(seed)
            self.idx = np.random.permutation(self.idx)

    def get(self, what, item):
        return self.sample.get(what, self.idx[item])

    def _build_tree(self, label="Shuffled", prefix=""):
        tree = Tree(prefix + label)
        subtree = self.sample._build_tree(label="SampleProvider")
        tree.add(subtree)
        return tree


class DictSampleProvider(SampleProvider):
    def __init__(self, context: Context, dictionary: dict, with_attributes: bool = False):
        super().__init__(context)
        self.with_attributes = with_attributes
        self._samples = {k: sample_factory(self.context, **v) for k, v in dictionary.items()}

    def __getattr__(self, key):
        if key in self._samples:
            return self._samples[key]
        raise AttributeError(f"{type(self).__name__} has no attribute '{key}'")

    def get(self, what, item):
        if item == 0:
            item = item + 1  # ✅✅ TODO provide the correct lenght
        return {k: v.get(what, item) for k, v in self._samples.items()}

    @property
    def range(self):
        start = max(s.range.start for s in self._samples.values())
        end = min(s.range.end for s in self._samples.values())
        return Range(start, end)

    def _build_tree(self, label="dict", prefix=""):
        tree = Tree(prefix + label)
        for k, v in self._samples.items():
            subtree = v._build_tree(prefix=f'"{k}" : ')
            tree.add(subtree)
        return tree


class Range:
    def __init__(self, start, end, step=1):
        # start and end are included
        self.start = start
        self.end = end
        self.step = step

    def __len__(self):
        return self.end - self.start

    def __repr__(self):
        return f"Range({self.start}, {self.end})"


class TimeDeltaShiftedSampleProvider(SampleProvider):
    def __init__(self, context: Context, timedelta: str, **kwargs):
        super().__init__(context)
        self.timedelta = frequency_to_timedelta(timedelta)
        self._sample = sample_factory(context, **kwargs)

    def compute_new_item(self, item, what=None):
        if item is None:
            return None
        return item + self.shift_item

    @property
    def range(self):
        start = max(self._sample.range.start + self.shift_item, 0)
        end = min(self._sample.range.end + self.shift_item, len(self._sample))
        return Range(start, end)

    @property
    def shift_item(self):
        # assert something here ?
        shift = self.timedelta // self._sample.frequency
        assert isinstance(shift, int), f"Shift must be an integer, got {shift} ({type(shift)})"
        return shift

    def get(self, what, item):
        new_item = self.compute_new_item(item, what)
        return self._sample.get(what, new_item)

    def _build_tree(self, prefix=""):
        txt = frequency_to_string(self.timedelta)
        if self.timedelta <= np.timedelta64(0, "s"):
            txt = f"[green]{txt}[/green]"
        else:
            txt = f"[red]{txt}[/red]"

        tree = Tree(f"{prefix} ⏱️  {txt}")
        subtree = self._sample._build_tree()
        tree.add(subtree)
        return tree


class GenericListSampleProvider(SampleProvider):
    def __init__(self, context: Context, tuple_: dict | list, timedeltas=None):
        super().__init__(context)
        if isinstance(tuple_, dict):
            if 'timedeltas' in tuple_:
                if timedeltas is not None:
                        raise ValueError(f"Duplicate value for timedeltas : {timedelta} vs {tuple_['timedelta']} ")
                timedeltas = tuple_.pop('timedeltas')

            new_tuple_ = []
            for timedelta in timedeltas:
                elt = tuple_.copy()
                if 'timedelta' in elt:
                    raise ValueError(f"Duplicate valye for timedelta and timedeltas")
                elt['timedelta'] = timedelta
                new_tuple_.append(elt)
            tuple_ = new_tuple_
        self._samples = tuple(sample_factory(context, **v) for v in tuple_)

    @property
    def range(self):
        start = max(s.range.start for s in self._samples)
        end = min(s.range.end for s in self._samples)
        return Range(start, end)

    def get(self, what, item):
        return tuple(v.get(what, item) for v in self._samples)

    #    def get(self, what, item):
    #        out = []
    #        for k, v in self._samples.items():
    #            k = frequency_to_timedelta(k)
    #            sample_step = k / v.frequency
    #            assert sample_step == int(sample_step)
    #            out.append(v.get(what, item + int(sample_step)))
    #        return out

    def _build_tree(self, label="GenericTuple", prefix=""):
        tree = Tree(prefix + label)
        for v in self._samples:
            subtree = v._build_tree()
            tree.add(subtree)
        return tree


class ListSampleProvider(GenericListSampleProvider):
    def _build_tree(self, label="tuple", prefix=""):
        tree = Tree(prefix + "🔗 " + label)
        for v in self._samples:
            subtree = v._build_tree()
            tree.add(subtree)
        return tree


class TensorSampleProvider(GenericListSampleProvider):
    def __init__(self, context: Context, tensor: dict, **kwargs):
        super().__init__(context, tuple_=tensor, **kwargs)

    def get(self, what, item):
        lst = super().get(what, item)
        assert isinstance(lst, (list, tuple)), f"Expected list or tuple, got {type(lst)}"
        return np.stack(tuple(lst))

    def _build_tree(self, label="Tensor", prefix=""):
        tree = Tree(prefix + "🔢 " + label)
        for v in self._samples:
            subtree = v._build_tree()
            tree.add(subtree)
        return tree


class Request(SampleProvider):
    def __init__(self, context: Context, variables: list[str], group: str):
        super().__init__(context)
        self.group = group
        self.variables = variables

    def _build_tree(self, label="Request", prefix=""):
        return Tree(f"{prefix}✉️  {label}({self.group}:{'/'.join(self.variables)})")

    def get(self, what, item):
        # this may be moved to the Mother class
        if isinstance(what, str):
            out = self._get(what, item)
            assert len(out) == 1, f"Expected single item for {what}, got {len(out)}"
            key = list(out.keys())[0]
            return out[key]
        return self._get(*what, item=item)

    @property
    def range(self):
        dh = DataHandler(self.context, self.group, None, variables=self.variables)
        return Range(0, len(dh))

    def _get(self, *what_and_item):
        *what, item = what_and_item
        dh = DataHandler(self.context, self.group, item, variables=self.variables)
        if item is not None:
            record = dh.record

        data = {}
        for w in what:
            if w == "__getitem__":
                data["data"] = record[self.group]
            elif w == "latitudes":
                data["latitudes"] = record.latitudes[self.group]
            elif w == "longitudes":
                data["longitudes"] = record.longitudes[self.group]
            elif w == "timedeltas":
                second = np.timedelta64(1, "s")
                data["timedeltas"] = record.timedeltas[self.group] // second
            elif w == "name_to_index":
                data["name_to_index"] = record.name_to_index[self.group]
            elif w == "statistics":
                data["statistics"] = record.statistics[self.group]
            elif w == "processors":
                processor_configs = self.context.data_config[self.group].get("processors", {})
                data["processors"] = [
                    [
                        n,
                        instantiate(
                            p,
                            name_to_index_training_input=record.name_to_index[self.group],
                            statistics=record.statistics[self.group],
                        ),
                    ]
                    for n, p in processor_configs.items()
                ]
            else:
                raise ValueError(f"Unknown request '{w}' for Request sample provider")
        return data


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

    def __len__(self):
        return len(self.ds)

    @property
    def record(self):
        if self._record is not None:
            return self._record
        self._record = self.ds[self.args]
        return self._record

    def __repr__(self):
        out = f"Request {self.dataset} @ {self.group} [{', '.join(self.variables)}]"
        if self.args is not None:
            out += f", args={self.args}"
        return out


def sample_factory(context, **kwargs):
    kwargs = kwargs.copy()
    if isinstance(context, dict):
        context = Context(**context)
    if context is None:
        context = Context()
    if "dictionary" in kwargs:
        return DictSampleProvider(context, **kwargs)
    if "tensor" in kwargs:
        return TensorSampleProvider(context, **kwargs)
    if "GROUPS" in kwargs:
        assert False, "GROUPS is deprecated, use dictionary instead"
    if "STEPS" in kwargs:
        assert False, "STEPS is deprecated, use tuple + timedelta instead"
    if "tuple" in kwargs:
        kwargs["tuple_"] = kwargs.pop("tuple")
        return ListSampleProvider(context, **kwargs)
    if "timedelta" in kwargs:
        return TimeDeltaShiftedSampleProvider(context, **kwargs)
    if "variables" in kwargs:
        return Request(context, variables=kwargs["variables"], group=kwargs["data"])
    assert False, f"Unknown sample type for kwargs {kwargs}"


# TEST ---------------------------------
if __name__ == "__main__":
    # user-friendly config would be:
    # fields=dict(
    #    steps=["-6h", "0h"],
    #    variables=["q_50", "2t"],
    #    data="era5",
    # ),
    yaml_str = """
data:
  era5:
    dataset:
      dataset: aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8
      set_group: era5
      # processors: ...
  snow:
    dataset: observations-testing-2018-2018-6h-v0
  metop_a:
    dataset: observations-testing-2018-2018-6h-v0

training_selection:
  # start=...
  end: "2018-11-01"

validation_selection:
  start: "2018-11-02"
  # end=...

sample:
  dictionary:
    input:
      dictionary:
        fields:
          tensor:
            - timedelta: "-6h"
              variables: ["q_50", "2t", "t_850"]
              data: era5
            - timedelta: "+12h"
              variables: ["q_50", "2t", "t_850"]
              data: era5
        fields2:
          tensor:
            timedeltas: ["-6h", "+12h"]
            variables: ["q_50", "2t", "t_850"]
            data: era5
        snow1:
          variables: ["sdepth_0"]
          data: snow
        snow2:
          timedelta: "+6h"
          variables: ["sdepth_0"]
          data: snow
        metop:
          tuple:
            - timedelta: "-12h"
              variables: ["scatss_1", "scatss_2"]
              data: metop_a
            - timedelta: "+6h"
              variables: ["scatss_1", "scatss_2"]
              data: metop_a
        #mixed:
        #  tuple:
        #    - dictionary:
        #        thing:
        #          variables: ["q_50"]
        #          data: era5
        #        another:
        #          variables: ["sdepth_0"]
        #          data: snow
        #    - variables: ["sdepth_0"]
        #      data: snow

        # user friendly config would be:
        # snow:
        #   timedeltas: ["0h", "+6h"]
        #   variables: ["sdepth_0"]
        #   data: snow
"""

    CONFIG = yaml.safe_load(yaml_str)

    def show_yaml(structure):
        return yaml.dump(structure, indent=2, sort_keys=False)

    def show_json(structure):
        return json.dumps(shorten_numpy(structure), indent=2)

    def shorten_numpy(structure):
        if isinstance(structure, np.ndarray):
            return f"np.array({structure.shape})"
        if isinstance(structure, (list, tuple)):
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

    # for x in [s, s.input, s.input.fields, s.input.metop, s.input.snow]:
    #    print()
    #    print("__len__:", len(x))
    #    print(x)

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
