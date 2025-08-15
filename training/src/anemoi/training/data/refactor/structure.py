# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import datetime
import inspect
import json
import os
from collections import defaultdict
from functools import wraps

import numpy as np
import yaml
from rich import print
from rich.console import Console
from rich.tree import Tree

from anemoi.training.data.refactor.sample_provider import sample_provider_factory
from anemoi.utils.dates import frequency_to_string


def format_shape(k, v):
    return f"shape: {v}"


def format_shorten(k, v):
    if len(str(v)) < 50:
        return f"{k}: {v}"
    return f"{k}: {str(v)[:50]}..."


def format_latlon(k, v):
    try:
        txt = f"[{np.min(v):.2f}, {np.max(v):.2f}]"
    except ValueError:
        txt = "[no np.min/np.max]"
    return f"{k}: [{txt}]"


def format_timedeltas(k, v):
    try:
        minimum = np.min(v)
        maximum = np.max(v)
        minimum = int(minimum)
        maximum = int(maximum)
        minimum = datetime.timedelta(seconds=minimum)
        maximum = datetime.timedelta(seconds=maximum)
        minimum = frequency_to_string(minimum)
        maximum = frequency_to_string(maximum)
    except ValueError:
        minimum = "no np.min"
        maximum = "no np.max"
    return f"timedeltas: [{minimum},{maximum}]"


def format_array(k, v):
    return f"{k} : array of shape {v.shape} with mean {np.nanmean(v):.2f}"


def format_data(k, v):
    if isinstance(v, np.ndarray):
        return format_array(k, v)
    return f"data: ❌ {type(v)}"


def format_default(k, v):
    if isinstance(v, (str, int, float, bool)):
        return f"{k} : {v}"
    if isinstance(v, (list, tuple)):
        return format_shorten(k, str(v))
    if isinstance(v, np.ndarray):
        return format_array(k, v)
    return f"{k} : {v.__class__.__name__}"


def format_none(k, v):
    return None


FORMATTERS = defaultdict(lambda: format_default)
FORMATTERS.update(
    dict(
        shape=format_shape,
        statistics=format_shorten,
        latitudes=format_latlon,
        longitudes=format_latlon,
        timedeltas=format_timedeltas,
        data=format_data,
        dataspecs=format_none,
    ),
)


def format_key_value(key, v):
    # TODO: should read from utils.configs
    if os.environ.get("ANEMOI_CONFIG_VERBOSE_STRUCTURE", "0") == "1":
        return FORMATTERS[key](key, v)
    return None


class StructureMixin:
    def __repr__(self, **kwargs):
        console = Console(record=True, width=120)
        tree = self.tree(**kwargs)
        with console.capture() as capture:
            console.print(tree)
        return capture.get()

    def add_from_native(self, **kwargs):
        other = _structure_factory(dataspecs=self.dataspecs, **kwargs)
        self.update(other)

    def format_native(self, **kwargs):
        return _structure_factory(dataspecs=self.dataspecs, **kwargs)

    def merge(self, other):
        merged = {}
        return _structure_factory(dataspecs=self.dataspecs, **merged)


Structure = StructureMixin


class TupleStructure(StructureMixin, tuple):
    # beware, inheriting from tuple, do not use __init__ method
    def tree(self, prefix="", **kwargs):
        tree = Tree(prefix + "🔗")
        for v in self:
            tree.add(v.tree(**kwargs))
        return tree

    def apply(self, func):
        return TupleStructure([x.apply(func) for x in self])

    def apply_to_self(self, func, **kwargs):
        return TupleStructure([x.apply_to_self(func, **kwargs) for x in self])

    def __call__(self, structure, **kwargs):
        assert isinstance(structure, TupleStructure), f"Expected TupleStructure, got {type(structure)}: {structure}"
        return TupleStructure(func(elt, **kwargs) for func, elt in zip(self, structure))

    def content(self, args):
        return [x.content(args) for x in self]

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            return super().__getattr__(name)
        return [getattr(x, name) for x in self]

    def _as_native(self):
        return tuple(x._as_native() for x in self)

    def is_compatible(self, other):
        if not isinstance(other, TupleStructure):
            return False
        if len(self) != len(other):
            return False
        for a, b in zip(self, other):
            if not a.is_compatible(b):
                return False
        return True

    def update(self, other_tuple_structure):
        if not isinstance(other_tuple_structure, TupleStructure):
            raise ValueError(f"Expected TupleStructure, got {type(other_tuple_structure)}: {other_tuple_structure}")
        if len(self) != len(other_tuple_structure):
            raise ValueError(f"Length mismatch: {len(self)} vs {len(other_tuple_structure)}")
        for a, b in zip(self, other_tuple_structure):
            a.update(b)

    def remove(self, *keys):
        for x in self:
            x.remove(*keys)


class DictStructure(StructureMixin, dict):

    def tree(self, prefix="", **kwargs):
        tree = Tree(prefix + "📖")
        for k, v in self.items():
            tree.add(v.tree(prefix=f"{k} : ", **kwargs))
        return tree

    def apply(self, func):
        return DictStructure({k: v.apply(func) for k, v in self.items()})

    def apply_to_self(self, func, **kwargs):
        return DictStructure({k: v.apply_to_self(func, **kwargs) for k, v in self.items()})

    def content(self, args):
        return {k: v.content(args) for k, v in self.items()}

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            return super().__getattr__(name)
        return {k: getattr(v, name) for k, v in self.items()}

    def __call__(self, structure, **kwargs):
        assert isinstance(structure, DictStructure), f"Expected DictStructure, got {type(structure)}: {structure}"
        assert set(self.keys()) == set(structure.keys()), f"Keys do not match: {self.keys()} vs {structure.keys()}"
        return DictStructure({k: self[k](structure[k], **kwargs) for k in self.keys()})

    def _as_native(self):
        return {k: v._as_native() for k, v in self.items()}

    def is_compatible(self, other):
        if not isinstance(other, DictStructure):
            return False
        if set(self.keys()) != set(other.keys()):
            return False
        for k in self.keys():
            if not self[k].is_compatible(other[k]):
                return False
        return True

    def update(self, other_dict_structure):
        if not isinstance(other_dict_structure, DictStructure):
            raise ValueError(f"Expected DictStructure, got {type(other_dict_structure)}: {other_dict_structure}")
        if not set(other_dict_structure.keys()) == set(self.keys()):
            raise ValueError(f"Keys do not match: {self.keys()} vs {other_dict_structure.keys()}")
        for k, v in self.items():
            v.update(other_dict_structure[k])

    def remove(self, *keys):
        for k, v in self.items():
            v.remove(*keys)


class LeafStructure(StructureMixin):
    """A final leave in the structure, contains always dataspecs
    and hopefully some useful content.
    """

    def __init__(self, **content):
        self._content = content
        self._names = list(self._content.keys())

    def content(self, args):
        if isinstance(args, str):

            return self._content[args]
        return self.__class__({k: self._content[k] for k in args})

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            return super().__getattr__(name)
        return self._content[name]

    def is_compatible(self, other_leaf_structure):
        if not isinstance(other_leaf_structure, self.__class__):
            return False
        if self._content["dataspecs"] != other_leaf_structure._content["dataspecs"]:
            return False

    def update(self, other_leaf_structure):
        if not isinstance(other_leaf_structure, self.__class__):
            raise ValueError(
                f"Expected {self.__class__.__name__}, got {type(other_leaf_structure)}: {other_leaf_structure}",
            )
        if self._content["dataspecs"] != other_leaf_structure._content["dataspecs"]:
            raise ValueError(
                f"Dataspecs do not match: {self._content['dataspecs']} vs {other_leaf_structure._content['dataspecs']}",
            )

        for k, v in other_leaf_structure._content.items():
            if k == "dataspecs":
                continue
            if k in self._content:
                raise ValueError(
                    f"Key {k} already exists, overwriting is not allowed yet. {self._names} vs {other_leaf_structure._names}",
                )
            self._content[k] = v

    def remove(self, *keys):
        for key in keys:
            if key not in self._content:
                raise KeyError(f"Key {key} not found in {self._content['dataspecs']}. Available: {self._names}")
            del self._content[key]
            self._names = [k for k in self._names if k != key]

    def tree(self, prefix="", verbose=False, **kwargs):

        if len(self._names) == 2:
            key = (set(self._names) - {"dataspecs"}).pop()
            v = self._content[key]
            txt = format_key_value(key, v)
            if txt is None:
                txt = type(v)
            return Tree(f"{prefix} 📦 {key}:{txt}")

        content_txt = " ".join(f"{k}" for k in self._content if k != "dataspecs")
        tree = Tree(f"{prefix} 📦 {content_txt}")
        for k, v in self._content.items():
            txt = format_key_value(k, v)
            if txt is not None:
                tree.add(txt)
        return tree

    def apply(self, func):
        if isinstance(func, str):
            if func not in self._content:
                raise ValueError(f"Function {func} not found in {self._content['dataspecs']}. Available: {self._names}")
            func = self._content[func]
        if not callable(func):
            raise ValueError(f"Expected a callable function, got {type(func)}: {func}")
        name = func.__name__
        print(func, name)
        new = func(**{k: self._content[k] for k in self._names})
        return self.__class__(**{"dataspecs": self._content["dataspecs"], name: new})

    def apply_to_self(self, func, output, merge=False, **kwargs):
        """Apply a function to the content of the structure."""
        if not callable(func):
            raise ValueError(f"Expected a callable function, got {type(func)}: {func}")

        # Use introspection to call func with only the arguments it accepts
        sig = inspect.signature(func)
        params = sig.parameters
        # Exclude 'self' if present
        arg_names = [
            name for name, param in params.items() if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        ]
        selected_content = {}
        for a in arg_names:
            if a not in self._content:
                raise ValueError(
                    f"Function requests argument {a}, but it's not found in {self._content['dataspecs']}. Available: {self._names}",
                )
            selected_content[a] = self._content[a]
        result = func(**selected_content)

        if output is dict or output is None:
            new = result
        elif isinstance(output, str):
            new = {output: result}
        elif isinstance(output, (list, tuple)):
            if len(output) != len(result):
                raise ValueError(
                    f"Output length {len(output)} does not match result length {len(result)} for {self._content['dataspecs']}.",
                )
            new = {name: result for name, result in zip(output, result)}
        else:
            raise ValueError(f"Unknown output type {type(output)}: {output}")

        if merge:
            return _structure_factory(**self._content, **new)
        return _structure_factory(dataspecs=self._content["dataspecs"], **new)

    def __call__(self, structure, function=None, input=None, result=None, **kwargs):
        assert isinstance(structure, LeafStructure), f"Expected LeafStructure, got {type(structure)}: {structure}"

        input_name = input or "data"
        result_name = result or input_name

        func = self._content[function]

        if not callable(func):
            raise ValueError(f"Expected a callable function in {self._content['dataspecs']}, got {type(func)}: {func}")

        x = structure.content(input_name)
        y = func(x)

        return structure.__class__(**{"dataspecs": structure.dataspecs, result_name: y})

    def _as_native(self, key=None):
        if key is not None:
            return self._content.get(key, None)
        return self._content


def on_structure(*args, output=None, **kwargs):
    def _decorator(func):
        @wraps(func)
        def wrapper(structure):
            return structure.apply_to_self(func, output=output, **kwargs)

        return wrapper

    if len(args) == 1 and callable(args[0]) and output is None:
        func = args[0]
        return _decorator(func)

    return _decorator


def structure_factory(content=None, dataspecs=None, native=None):
    if content is not None:
        if dataspecs is not None or native is not None:
            raise ValueError("Cannot provide both content and dataspecs/native.")
        return _structure_factory(**content)
    return _structure_factory(dataspecs=dataspecs, native=native)


def _structure_factory(**content):
    check_structure(**content)
    dataspecs = content["dataspecs"]

    if isinstance(dataspecs, str) and dataspecs.endswith(".tensor"):
        return LeafStructure(**content)

    if isinstance(dataspecs, (list, tuple)):
        lst = []
        for i in range(len(dataspecs)):
            lst.append(_structure_factory(**{key: content[key][i] for key in content}))
        return TupleStructure(lst)

    assert isinstance(dataspecs, dict), "Expected dicts"
    dic = {}
    for k in dataspecs.keys():
        dic[k] = _structure_factory(**{key: content[key][k] for key in content})
    return DictStructure(dic)


def check_structure(**content):
    assert "dataspecs" in content, f"Missing 'dataspecs' in content. Found only {list(content.keys())}"

    dataspecs = content["dataspecs"]
    for v in content.values():
        if isinstance(dataspecs, str) and dataspecs.endswith(".tensor"):
            continue

        if isinstance(dataspecs, dict):
            assert isinstance(
                v,
                dict,
            ), f"Expected all values to be dict, got {type(v)} != {type(dataspecs)} whith {v} and {dataspecs}"
            assert set(v.keys()) == set(
                dataspecs.keys(),
            ), f"Expected the same keys, got {list(v.keys())} vs. {list(dataspecs.keys())}"

        if isinstance(dataspecs, (list, tuple)):
            assert isinstance(
                v,
                (list, tuple),
            ), f"Expected all values to be lists or tuples, got {type(v)} != {type(dataspecs)} whith {v} and {dataspecs}"
            assert len(v) == len(dataspecs), f"Expected the same length as first, got ✅{v}✅ vs ❌{dataspecs}❌"


def test():
    yaml_str = """
sources:
  training:
    era5:
      dataset:
        dataset: aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8
        set_group: era5
    snow:
      dataset: observations-testing-2018-2018-6h-v0
    metop_a:
      dataset: observations-testing-2018-2018-6h-v0
      normaliser:
            "scatss_1": "mean-std"
            "scatss_2": "min-max"
            "scatss_3": {"name": "custom-normaliser", "theta": 0.5, "rho": 0.1}
      imputer:
            "scatss_1": special
            "scatss_2": other
            "scatss_3": {"name": "custom-imputer", "theta": 0.5, "rho": 0.1}
      extra:
        user_key_1: a
        user_key_2:
            1: foo
            2: bar



training_selection:
  # start=...
  end: "2018-11-01"

validation_selection:
  start: "2018-11-02"
  # end=...


sample:
   use_case: "downscaling"
   high_res: ......

sample:
      dictionary:
        ex_simple_tensor:
          tensor:
            - variables: ["metop_a.scatss_1", "metop_a.scatss_2"]

        # not supported
        ex_simple_tensor_shortcut:
          variables: ["metop_a.scatss_1", "metop_a.scatss_2"]

        ex_simple_dict:
          dictionary:
            key1:
                tensor:
                  - variables: ["snow.stalt", "snow.sdepth_0"]
            key2:
                tensor:
                  - variables: ["metop_a.scatss_1", "metop_a.scatss_2"]

        ex_simple_offset:
          tensor:
            - variables: ["metop_a.scatss_1", "metop_a.scatss_2"]
              offset: "-12h"

        ex_simple_offset_also:
          offset: "-12h"
          tensor:
            - variables: ["metop_a.scatss_1", "metop_a.scatss_2"]

        ex_adding_offsets:
          offset: "-12h"
          tensor:
            - variables: ["metop_a.scatss_1", "metop_a.scatss_2"]
              offset: "-6h"

        ex_dict:
          dictionary:
            key1:
                offset: "-6h"
                tensor:
                    - variables: ["snow.stalt", "snow.sdepth_0"]
            key2:
                offset: "0h"
                tensor:
                   - variables: ["metop_a.scatss_1", "metop_a.scatss_2"]

        ex_tensor_2:
          tensor:
            - offset: ["-6h", "0h", "+6h"]
            - variables: ["era5.2t", "era5.10u"]

        # choose the order of dimensions in the tensor
        ex_tensor_3:
          tensor:
            - variables: ["era5.2t", "era5.10u"]
            - offset: ["-6h", "0h", "+6h"]

        # this would fail, as obs are not regular:
        # ex_tensor_failing:
        #   tensor:
        #     - variables: ["metop_a.scatss_1", "metop_a.scatss_2", "snow.sdepth_0"]

        # do this instead when the tensors are not regular and get a tuple of tensors:
        ex_tuple:
          tuple:
            loop:
              - offset: ["-6h", "0h", "+6h"]
            template:
              tensor:
                - variables: ["metop_a.scatss_1", "metop_a.scatss_2"]

        #test_offset4:
        #  offset: "-6h"
        #  structure:
        #    offset: "-6h"
        #    structure:
        #      variables: ["metop_a.scatss_1", "metop_a.scatss_2"]
        #
"""

    import sys

    if len(sys.argv) > 1:

        path = yaml.safe_load(sys.argv[1])
        with open(path) as f:
            yaml_str = f.read()
        CONFIG = yaml.safe_load(yaml_str)
        sample_config = CONFIG["sample"]
        sources_config = CONFIG["data"]

    else:
        CONFIG = yaml.safe_load(yaml_str)
        sample_config = CONFIG["sample"]
        sources_config = CONFIG["sources"]["training"]
    print(sources_config)

    def show_yaml(structure):
        return yaml.dump(structure, indent=2, sort_keys=False)

    def show_json(structure):
        return json.dumps(shorten_numpy(structure), indent=2)

    def shorten_numpy(structure):
        from anemoi.training.data.refactor.data_handler import DataHandler

        if isinstance(structure, np.ndarray):
            if np.issubdtype(structure.dtype, np.floating):
                return f"np.array{structure.shape} with mean {np.nanmean(structure):.2f}"
            return f"np.array{structure.shape} with mean {np.nanmean(structure)}"
        if isinstance(structure, (list, tuple)):
            if structure and all(isinstance(item, int) for item in structure):
                return "[" + ", ".join(map(str, structure)) + "]"
            return [shorten_numpy(item) for item in structure]
        if isinstance(structure, dict):
            return {k: shorten_numpy(v) for k, v in structure.items()}
        if isinstance(structure, DataHandler):
            return str(structure)
        return structure

    training_context = dict(
        sources=sources_config,
        start=None,
        end=None,
        frequency="6h",
    )
    if True:
        # if False:

        print("✅✅  --------")
        for key, config in sample_config["dictionary"].items():
            print(f"[yellow]- {key} : getting data [/yellow]")
            print(yaml.dump(config, indent=2, sort_keys=False))
            s = sample_provider_factory(**training_context, **config)
            print(s)
            print("length : ", len(s))
            name_to_index = s.name_to_index
            print(f"name_to_index = {name_to_index}")
            statistitics = s.statistics
            print(f"statistics = {statistitics}")
            print("sp[1] = ", show_json(s[1]))

    print("............................")

    i = 1

    config = """dictionary:
        fields:
          tensor:
            - variables: ["era5.2t", "era5.10u", "era5.10v"]
            - offset: ["-6h"]
        other_fields:
          tensor:
            - offset: ["-6h", "+6h"]
            - variables: ["era5.2t", "era5.10u"]
        observations:
          tuple:
            loop:
              - offset: ["-6h", "0h", "+6h"]
            template:
              tensor:
                - variables: ["metop_a.scatss_1", "metop_a.scatss_2"]
    """
    config = yaml.safe_load(config)
    sp = sample_provider_factory(**training_context, **config)

    content = {
        "name_to_index": sp.name_to_index,
        "statistics": sp.statistics,
        "extra": sp.extra,
        "dataspecs": sp.dataspecs,
        "normaliser": sp.normaliser,
        # "shape": sp.shape,
        # "data_spec": sp.data_spec,
        **sp[1],
    }
    print("info_from_sample_provider", content)

    data = sp[i]
    print("Native types data : ")
    for k, v in data.items():
        print(f"data['{k}'] = {show_json(v)}")

    obj = _structure_factory(**content)
    print("Data as object :")
    print(obj)
    print(f"{obj['fields'].name_to_index=}")
    print(f"{obj['fields'].normaliser=}")
    print(f"{obj['fields'].extra=}")
    print(f"{obj['fields'].statistics=}")
    print(f"{obj['fields'].dataspecs=}")
    print(f"{obj['observations'][0].statistics=}")
    print(f"{obj['observations'][0].data=}")
    print(f"{obj['observations'][0].dataspecs=}")

    def my_function(name_to_index, statistics, normaliser, **kwargs):
        return f"Normalisers build from: {name_to_index=}, {statistics=}, {normaliser=}"

    result = obj.apply(my_function)
    print(result)
    print()
    print(f"{result['observations'][0].my_function=}")
    print(f"{result['fields'].my_function=}")
    print()
    print(f"{result['fields']=}")
    print(f"{result['fields']._as_native()}")
    print(f"{result['fields']._as_native('my_function')=}")

    print(f"{str(sp.get_native(2))[:1000]=}")
    print(f"{sp.get_obj(2)=}")
    # print(sp.get_obj(2).__repr__(verbose=True))

    print("[blue]Result of applying the function to the structure:[/blue]")

    @on_structure(output="new_content")
    def times100(data, name_to_index, **kwargs):
        # name_to_index is not used here, but could be useful in a real function
        return data * 100

    print(times100(obj))

    # @on_structure(output=["new1", "new2"])
    # def times100_(data, name_to_index, **kwargs):
    #     # name_to_index is not used here, but could be useful in a real function
    #     return data * 100, data * 10
    # print(times100_(obj))
    # obj.update(function_structure)
    # print("Updated object with function structure:")
    # print(obj)
    # obj.apply("doubler")

    print("------------------------")
    obj.remove("extra", "normaliser", "latitudes", "longitudes", "timedeltas")

    native = times100(obj).new_content
    try:
        print(native.new_content)
        assert False, "Expected new_content not an object with attributes."
    except AttributeError:
        pass

    print(f"{obj.format_native(new=native)=}")

    print("------------------------")

    @on_structure
    def times_100(data, name_to_index, **kwargs):
        # name_to_index is not used here, but could be useful in a real function
        return dict(new_content=data - 100)

    print(times_100(obj))


if __name__ == "__main__":
    test()
