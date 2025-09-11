# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from functools import wraps
from typing import Union

import boltons
import numpy as np
import torch
import yaml
from boltons.iterutils import default_enter as _default_enter
from boltons.iterutils import default_exit as _default_exit
from boltons.iterutils import get_path as _get_path
from boltons.iterutils import remap as _remap
from boltons.iterutils import research as _research  # noqa: F401
from rich import print

from anemoi.training.data.refactor.formatting import to_str
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
NestedTensor = Union[
    torch.Tensor,
    Sequence["NestedTensor"],  # list/tuple of NestedTensor
    Mapping[str | int, "NestedTensor"],  # dict with str/int keys
]


def make_schema(structure):
    """Return a fully nested schema from any structure."""

    def element_description(element):
        return dict(type=type(element).__name__, content=None)

    if is_box(structure):
        content = {k: element_description(v) for k, v in structure.items()}
        content["_anemoi_schema"] = True
        return {"type": "box", "content": content, "_anemoi_schema": True}
    if isinstance(structure, dict):
        return {"type": "dict", "content": {k: make_schema(v) for k, v in structure.items()}, "_anemoi_schema": True}
    if isinstance(structure, (list, tuple)):
        return {"type": "tuple", "content": [make_schema(v) for v in structure], "_anemoi_schema": True}
    assert False, f"Unknown structure type: {type(structure)}"


class Dict(dict):
    def copy(self):
        return self.__class__(self)

    def __copy__(self):
        return self.__class__(self)

    def __deepcopy__(self, memo):
        import copy

        return self.__class__(copy.deepcopy(dict(self), memo))


class Box(Dict):
    # Flag a dict as a box
    def __repr__(self):
        return to_str(self, name=" ")

    def to_str(self, name):
        return to_str(self, name=name)


class Tree(Dict):
    def __repr__(self):
        return to_str(self, name=" ")

    def to_str(self, name):
        return to_str(self, name=name)


def cast_output_to_box(func, check_dict_output=True):
    @wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        if not isinstance(res, dict) and check_dict_output:
            raise ValueError(f"Function {func.__name__} did not return a dict, got {type(res)}")
        if isinstance(res, dict):
            res = Box(res)
        return res

    return wrapper


class Batch(Tree):
    def __getattr__(self, name):
        if name not in self:
            raise AttributeError(f"'Batch' object has no attribute '{name}'")
        value = self[name]
        if isinstance(value, dict) and not isinstance(value, Box):
            value = Batch(value)
        return value

    def create_function(self, func, *args, check_dict_output=True, **kwargs):
        tree_of_functions = self.box_to_any(func, *args, **kwargs)
        return Function(tree_of_functions)

    def create_module_dict(self, constructor, *args, **kwargs):
        # it is expected that 'constructor' creates a nn.Module
        tree_of_modules = self.box_to_any(constructor, *args, **kwargs)
        return as_module_dict(tree_of_modules)

    def box_to_box(self, func, *args, **kwargs):
        def transform(path, key, value):
            if not isinstance(value, Box):
                return key, value
            # kwargs overwride the value content
            box = {k: v for k, v in value.items() if k not in kwargs}
            res = func(*args, **box, **kwargs)
            return key, Box(res)

        return _remap(self, visit=transform, enter=_stop_if_box_enter, exit=ensure_output_is_box_exit)

    def box_to_any(self, func, *args, **kwargs):
        def transform(path, key, value):
            if not isinstance(value, Box):
                return key, value
            # kwargs overwride the value content
            box = Box({k: v for k, v in value.items() if k not in kwargs})
            return key, func(*args, **box, **kwargs)

        return _remap(self, visit=transform, enter=_stop_if_box_enter, exit=ensure_output_is_box_exit)

    def pop_content(self, key):
        return self.__class__(remove_content(self, key))

    def select_content(self, *keys):
        return self.__class__(select_content(self, *keys))

    def merge_content(self, other):
        return self.__class__(merge_boxes(self, other))

    def unwrap(self, *requested_keys):
        return self.__class__(unwrap(*requested_keys))


def as_module_dict(structure):
    """Transform a structure into a nested MuduleDict.
    The leaves of the scructure must be nn.Module.
    """

    def to_module_dict_exit(path, key, old_parent, new_parent, new_items):
        res = _default_exit(path, key, old_parent, new_parent, new_items)
        if isinstance(old_parent, dict):
            res = ModuleDictt(res)
        if isinstance(old_parent, Box):
            assert False, (path, key, old_parent, new_parent, new_items)
        # if isinstance(old_parent, list):
        #    res = torch.nn.ModuleList(res)
        # if isinstance(old_parent, tuple):
        #    res = tuple(res)
        assert isinstance(res, torch.nn.Module), f"Expected nn.Module, got {type(res)} at {path}, {key}"
        return res

    return _remap(structure, exit=to_module_dict_exit, enter=_stop_if_box_enter)


class ModuleDictt(torch.nn.ModuleDict):
    def __call__(self, other, *args, _output_box=True, _output="data", **kwargs):
        def apply(current, other_current, path):

            if isinstance(current, torch.nn.ModuleDict) or isinstance(current, torch.nn.ModuleList):
                res = Batch()
                for k in current.keys():
                    if k not in other_current:
                        raise ValueError(f"Key {k} not found in at {path}. In {other}")
                    res[k] = apply(current[k], other_current[k], path + (k,))
                return res

            if not isinstance(current, torch.nn.Module):
                raise ValueError(f"Expected nn.Module at {path}, got {type(current)}")

            res = current(*args, **kwargs, **other_current)
            if _output:
                res = {**other_current, _output: res}

            if _output_box:
                if not isinstance(res, dict):
                    raise ValueError(f"Function at {path} did not return a dict, got {type(res)}")
                res = Box(res)

            return res

        return apply(self, other, ())


class Function(Tree):
    emoji = "()"

    def __call__(self, other, *args, _output_box=True, **kwargs):
        def apply(path, key, func):
            if not callable(func):
                return key, func

            x = _get_path(other, path + (key,), default=None)
            if not x:
                raise ValueError(f"Box not found in at {path}, {key=}. In {other}")
            res = func(*args, **kwargs, **x)

            if _output_box:
                if not isinstance(res, dict):
                    raise ValueError(f"Function at {path}, {key} did not return a dict, got {type(res)}")
                res = Box(res)

            return key, res

        res = _remap(self, visit=apply)
        return Batch(res)

    def apply_on_multiple_structures(self, *args, _output_box=True, **kwargs):

        def apply(path, key, func):
            if not callable(func):
                return key, func

            def search(x):
                return _get_path(x, path + (key,), default=None)

            args = [search(a) if isinstance(a, Batch) else a for a in args]
            kwargs = {k: search(v) if isinstance(v, Batch) else v for k, v in kwargs.items()}

            res = func(*args, **kwargs)
            if _output_box:
                if not isinstance(res, dict):
                    raise ValueError(f"Function at {path}, {key} did not return a dict, got {type(res)}")
                res = Box(res)
            return res

        res = _remap(self, visit=apply)
        return Batch(res)

    def to_str(self, name):
        return to_str(self, name=name + self.emoji, _boxed=False)

    def __repr__(self):
        return to_str(self, name=self.emoji, _boxed=False)


def _check_key_in_dict(dic, key):
    if not isinstance(dic, dict):
        raise ValueError(f"Expected dict, got {type(dic)}")
    if key not in dic:
        raise ValueError(f"Key {key} not found in dict. Available keys are: {list(dic.keys())}")


def is_schema(x):
    if not isinstance(x, dict):
        return False
    return "_anemoi_schema" in x


def is_final(structure):
    return not isinstance(structure, (dict, list, tuple))


def is_box(structure):
    if isinstance(structure, Box):
        return True
    if not isinstance(structure, dict):
        return False
    # Not so reliable, see inside the dict if there is a final element
    return any(is_final(v) for v in structure.values())


def merge_boxes(*structs, overwrite=True):
    """Merge multiple structures
    >>> a = {"foo": [{"x": 1, "y": 2}, {"a": 3}]}
    >>> b = {"foo": [{"z": 10}, {"b": 4}]}
    >>> merge_boxes(a, b)
    {'foo': [{'x': 1, 'y': 2, 'z': 10}, {'a': 3, 'b': 4}]}
    >>> c = {"foo": [{"z": 10}, {"a": 999}]}
    >>> merge_boxes(a, c)
    {'foo': [{'x': 1, 'y': 2, 'z': 10}, {'a': 3, 'b': 4}]}
    """

    def exit(path, key, old_parent, new_parent, new_items):
        if not is_box(old_parent):
            return _default_exit(path, key, old_parent, new_parent, new_items)
        # If we reach here, it means old_parent is a box
        # We need to merge new_items into the corresponding box in old_parent
        res = Box()
        for struct in structs:
            box = _get_path(struct, path + (key,), default={})
            if not overwrite and set(box.keys()) & set(res.keys()):
                raise ValueError(f"Conflicting keys found in structures: {set(box.keys()) & set(res.keys())}")
            res.update(**box)
        return res

    return _remap(structs[0], exit=exit)


def wrap_in_box(**structures):
    if all(isinstance(v, dict) for v in structures.values()):
        first = next(iter(structures.values()))
        b = Batch()
        for key in first.keys():
            assert all(
                set(s.keys()) == set(first.keys()) for s in structures.values()
            ), "All dicts must have the same keys to wrap in box"
            values = {k: s[key] for k, s in structures.items()}
            b[key] = wrap_in_box(**values)
        return b
    return Box(**structures)


def _stop_if_box_enter(path, key, value):
    if is_box(value):
        return value, []
    return _default_enter(path, key, value)


def ensure_output_is_box_exit(path, key, old_parent, new_parent, new_items):

    res = _default_exit(path, key, old_parent, new_parent, new_items)
    if is_box(old_parent):
        res = Box(res)
    return res


def _for_each_expanded_box(fconfig_tree, func, *args, **kwargs):
    # apply the func to each box in the fconfig_tree tree
    is_callable = dict(callable=None)

    def transform(path, key, fconfig_box):
        if not is_box(fconfig_box):
            # apply only on boxes
            return key, fconfig_box
        if any(callable(v) for k, v in fconfig_box.items()):
            return key, fconfig_box

        # print(f"DEBUG: {path} {key}, applying function {func.__name__} to {value} {args=} {kwargs=}")
        value = func(**fconfig_box, **kwargs)
        # print(f"  -> {value=}")
        if isinstance(value, dict):
            value = Box(value)

        if is_callable["callable"] is None:
            is_callable["callable"] = callable(value)
        is_callable["callable"] = is_callable["callable"] and callable(value)
        return key, value

    structure = _remap(fconfig_tree, visit=transform, enter=_stop_if_box_enter, exit=ensure_output_is_box_exit)
    # if is_callable["callable"]:
    #    return nested_to_callable(structure)
    return structure


def apply(func, structure, *args, **kwargs):
    _made_callable = False
    if not callable(func):
        func = nested_to_callable(func)
        _made_callable = True
    try:
        return func(structure, *args, **kwargs)
    except Exception as e:
        e.add_note(f"While applying function to structure: {to_str(func, 'Function Tree')}. {_made_callable=}")
        raise ValueError(f"Error applying function to structure: {e}")


# decorator
def box_to_function(func, **options):
    if options or not callable(func):
        raise Exception("box_to_function decorator takes at most one non-keyword argument, the function to wrap.")

    @wraps(func)
    def wrapper(fconfig_n, *fargs, **fkwargs):
        res_n = _for_each_expanded_box(fconfig_n, func, *fargs, **fkwargs)
        return nested_to_callable(res_n)

    return wrapper


# decorator
def apply_to_box(func, **options):
    if options or not callable(func):
        raise Exception("apply_to_box decorator takes at most one non-keyword argument : the function to wrap.")

    @wraps(func)
    def wrapper(fconfig_n, *fargs, **fkwargs):
        return _for_each_expanded_box(fconfig_n, func, *fargs, **fkwargs)

    return wrapper


# decorator
def make_output_callable(func):
    """Func is a function which takes a nested structure and returns a nested structure where the leaves are functions
    This decorator adds one step where the output of this function is made callable
    """
    if not callable(func):
        raise Exception("make_output_callable decorator takes a callable argument, the function to wrap.")

    @wraps(func)
    def wrapper(*args, **kwargs):
        res_tree = func(*args, **kwargs)
        return nested_to_callable(res_tree)

    return wrapper


def nested_to_callable(f_tree):
    """The input is a nested structure where the leaves are functions
    The output is a callable as follow:

    The input of this callable must be a similar structure
    The output of this callable is a similar stucture where each function is applied to each box.
    """
    if callable(f_tree):
        print("DEBUG: already callable")
        return f_tree

    def function_on_tree(x_tree, *args, **kwargs):

        def apply(path, key, func):
            if not callable(func):
                # apply only on functions
                return key, func

            x = _get_path(x_tree, path + (key,), default=None)
            if not x:
                raise ValueError(f"Box not found in {x_tree} at {path}, {key=}")
            new_box = func(*args, **kwargs, **x)
            if not isinstance(new_box, dict):
                raise ValueError(f"Expected dict, but function returned {type(new_box)} in {path}, {key=}")
            return key, new_box

        return _remap(f_tree, visit=apply)

    function_on_tree._anemoi_function_str = to_str(f_tree, "(function)", _boxed=False)

    return function_on_tree


def remove_content(nested, key):
    def pop(path, k, box):
        if not is_box(box):
            return k, box
        _check_key_in_dict(box, key)
        v = box.pop(key)
        return k, Box({key: v})

    return _remap(nested, visit=pop, enter=_stop_if_box_enter)


def select_content(nested, *keys):

    def select(path, key, box):
        if not is_box(box):
            return key, box
        for k in keys:
            _check_key_in_dict(box, k)
        return key, Box({k: box[k] for k in keys if k in box})

    return _remap(nested, visit=select, enter=_stop_if_box_enter)


def unwrap(nested, *requested_keys):

    res = []
    for requested_key in requested_keys:

        def select(path, key, box):
            if not is_box(box):
                return key, box
            return key, box[requested_key]

        res.append(_remap(nested, visit=select, enter=_stop_if_box_enter))
    return res


def rearrange(mappings, sources, _add_origin=True):
    """Rearrange the boxes from sources according to the mapping.

    sources is a nested structure where the leaf are boxes

    mapping is a dict {target_path: source_path}
    where target_path and source_path are path to boxes
    source_path must be consitent with the sources provided

    """

    def _path_to_tuple(p):
        if isinstance(p, (list, tuple)):
            return p
        return tuple(int(x) if x.isdigit() else x for x in p.split("."))

    def resolve(path):
        _path = _path_to_tuple(path)
        try:
            box = _get_path(sources, _path)
        except boltons.iterutils.PathAccessError as e:
            if len(_path) > 1:
                container = _get_path(sources, _path[:-1], None)
                if isinstance(container, dict):
                    e.add_note(f"Available keys in container: {list(container.keys())}")
                if isinstance(container, list):
                    e.add_note(f"Container is a list of length {len(container)}")
            raise

        if _add_origin:
            box["_origin"] = path
        return box

    def insert_path_in_list(cur, path, value):
        p, *rest = path
        assert isinstance(p, int)

        while len(cur) <= p:
            cur.append(None)

        if not rest:  # last step
            cur[p] = value
            return

        if not isinstance(cur[p], (dict, list)):
            cur[p] = {} if isinstance(rest[0], str) else []

        insert_path(cur[p], rest, value)

    def insert_path_in_dict(cur, path, value):
        p, *rest = path
        assert isinstance(p, str)

        if not rest:  # last step
            cur[p] = value
            return

        if p not in cur:
            cur[p] = {} if isinstance(rest[0], str) else []

        insert_path(cur[p], rest, value)

    def insert_path(cur, path, value):
        p = path[0]
        if isinstance(p, int):
            if not isinstance(cur, list):
                raise TypeError(f"Expected list at {path}, got {type(cur).__name__}")
            insert_path_in_list(cur, path, value)
        else:
            if not isinstance(cur, dict):
                raise TypeError(f"Expected dict at {path}, got {type(cur).__name__}")
            insert_path_in_dict(cur, path, value)

    root = {}
    for k, v in mappings.items():
        v = resolve(v)
        insert_path(root, _path_to_tuple(k), v)
    return Batch(root)


def test_custom(path):
    path = yaml.safe_load(path)
    with open(path) as f:
        yaml_str = f.read()
    CONFIG = yaml.safe_load(yaml_str)
    sample_config = CONFIG["sample"]
    sources_config = CONFIG["data"]

    do_something_with_this


def test_one(training_context):
    from anemoi.training.data.refactor.sample_provider import sample_provider_factory

    cfg_1 = """dictionary:
                    lowres:
                        container:
                          data_group: "era5"
                          variables: ["2t", "10u", "10v"]
                          dimensions: ["values", "variables", "ensembles"]

                    highres:
                      for_each:
                        - offset: ["-6h", "0h", "+6h", "+12h", "+18h", "+24h"]
                        - container:
                            data_group: "era5"
                            variables: ["2t", "10u", "10v"]
                            dimensions: ["variables", "values"]
                      #rollout:
                      #  kind: prognostics
                      #  steps: [0h, +6h, +12h, +18h, +24h]

                      #  input: [-6h, 0h]
                      #  target: 6h


            """

    print("✅✅✅✅✅✅✅✅✅✅✅✅✅✅")
    config = yaml.safe_load(cfg_1)

    sp = sample_provider_factory(**training_context, **config)
    print(sp)
    schema = sp.dataschema

    # print(schema)
    # print(repr_schema(schema, "schema"))

    print("✅ sp.static_info", sp.static_info)

    data = sp[1]
    print(data)
    print(to_str(data, name="✅ Data"))

    data = sp.static_info.merge_content(data)
    print("✅ Data", data)

    def to_tensor(**kwargs):
        import torch

        def f(arr):
            return torch.Tensor(arr)

        res = {k: f(v) if k == "data" else v for k, v in kwargs.items()}
        res = Box(res)
        return res

    data = data.box_to_box(to_tensor)

    def to_gpu(**kwargs):
        def f(arr):
            return arr.to("cuda")

        res = {k: f(v) if k == "data" else v for k, v in kwargs.items()}
        # res = Box(res)
        return res

    new_data = data.box_to_box(to_gpu)
    print("✅ Data on GPU", new_data)
    print(type(new_data["lowres"]))

    guessed = make_schema(data)
    print(to_str(schema, "Actual Schema"))
    print(to_str(guessed, "Actual Schema"))

    extra = data.pop_content("extra")
    print("Data after pop_content extra", data)
    print("Extra after pop_content extra", extra)

    lat_lon = data.select_content("latitudes", "longitudes")
    print("Selected lat_lon", lat_lon)

    def build_normaliser(statistics, **kwargs):
        print("✅💬 Building from", statistics)
        mean = np.mean(statistics["mean"])

        def func(data, **kwargs):
            new = data - mean
            return dict(data=new, **kwargs)

        # norm = Normaliser(mean,...)
        # def f(x):
        #    return norm.transform(x["data"])

        return func

    normaliser = sp.static_info.create_function(build_normaliser)

    print("Normaliser function", normaliser)
    n_data = normaliser(data)

    d = data.select_content("data")
    n = n_data.select_content("data")
    print_columns(d.to_str("Unnormalised data"), n.to_str("Normalised data"))


from rich.console import Console
from rich.table import Table


def print_columns(*args):

    console = Console()
    table = Table(show_header=False, box=None)
    for a in args:
        table.add_column()
    table.add_row(*args)
    console.print(table)


def test_two(training_context):

    from anemoi.training.data.refactor.sample_provider import sample_provider_factory

    cfg_2 = """dictionary:
                  prognostics:
                    for_each:
                      - offset: ["-6h", "0h", "+6h", "+12h", "+18h"]
                      - container:
                          dimensions: ["values", "variables"]
                          variables: ["2t", "10u", "10v"]
                          data_group: "era5"
                  #forcings:
                  #  for_each:
                  #    - offset: ["-6h", "0h", "+6h", "+12h", "+18h"]
                  #    - tensor:
                  #        - ensembles: False
                  #        - values: True
                  #        - variables: ["era5.2t", "era5.10u", "era5.10v"]
                  #diagnostics:
                  #  for_each:
                  #    - offset: ["-6h", "0h", "+6h", "+12h", "+18h"]
                  #    - tensor:
                  #        - ensembles: False
                  #        - values: True
                  #        - variables: ["era5.2t", "era5.10u", "era5.10v"]
                  #prognostics_tuple:
                  #  tuple:
                  #    for_each:
                  #      - offset: ["-6h", "0h", "+6h", "+12h", "+18h"]
                  #    template:
                  #      tensor:
                  #        - ensembles: False
                  #        - values: True
                  #        - variables: ["era5.2t", "era5.10u", "era5.10v"]

                  #observations:
                  #  tuple:
                  #    for_each:
                  #      - offset: ["-6h", "0h", "+6h"]
                  #    template:
                  #      tensor:
                  #        - variables: ["metop_a.scatss_1", "metop_a.scatss_2"]
            """
    config = yaml.safe_load(cfg_2)
    sp = sample_provider_factory(**training_context, **config)
    print("static_info", sp.static_info)
    data = sp[1]
    data = sp.static_info.merge_content(data)
    print("Full Data", data)

    def i_to_delta(i, frequency):
        frequency = frequency_to_timedelta(frequency)
        delta = frequency * i
        sign = "+" if i > 0 else ""
        if delta:
            return sign + frequency_to_string(delta)
        return "0h"

    data = data.select_content("data", "_offset")

    print("[red] ✅ Rollout for prognostic data[/red]")
    # TODO: same logic for Forcings data, and for Diagnostic data
    rollout_config = []
    n_rollout = 3
    n_step_input = 2  # TODO: use this below to create the config
    frequency = "6h"
    for i in range(n_rollout):
        rollout_config.append(
            {
                "input.prognostics.0": "previous_input.prognostics.1",
                "input.prognostics.1": "output.prognostics",
                "target.prognostics": f"from_dataset.prognostics.{i_to_delta(i + 1, frequency)}",
            },
        )
    rollout_config[0]["input.prognostics.0"] = f"from_dataset.prognostics.{i_to_delta(-1, frequency)}"
    rollout_config[0]["input.prognostics.1"] = f"from_dataset.prognostics.{i_to_delta(0, frequency)}"

    rollout_config[1]["input.prognostics.0"] = f"from_dataset.prognostics.{i_to_delta(0, frequency)}"

    print(yaml.dump(dict(rollout=rollout_config), sort_keys=False))
    from_dataset = data
    previous_input = {}
    output = {}
    for i in [0, 1, 2]:
        print(".............")
        cfg = rollout_config[i]
        print(cfg)
        sources = dict(from_dataset=from_dataset, previous_input=previous_input, output=output)
        print(f"Building data for rollout = {i}")
        print_columns(
            to_str(sources["from_dataset"], "from_dataset"),
            to_str(sources["previous_input"], "previous_input"),
            to_str(sources["output"], "output"),
        )
        input_target = rearrange(cfg, sources)
        print()
        input = input_target.input
        target = input_target.target
        print_columns(input.to_str(f"input({i})"), target.to_str(f"target({i})"))

        output = input_target["target"]
        previous_input = input_target["input"]


def test_three(training_context):
    from anemoi.training.data.refactor.sample_provider import sample_provider_factory

    cfg_3 = """dictionary:
                  ams:
                    for_each:
                      - offset: ["-6h", "0h", "+6h"]
                      - container:
                          variables: ["scatss_1", "scatss_2"]
                          data_group: "metop_a"
            """
    config = yaml.safe_load(cfg_3)
    sp = sample_provider_factory(**training_context, **config)
    print(sp)


def test():

    import sys

    if len(sys.argv) > 1 and not sys.argv[1].isdigit():
        return test_custom(sys.argv[1])

    source_yaml = """sources:
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
    """
    sources_config = yaml.safe_load(source_yaml)["sources"]["training"]
    print(sources_config)

    training_context = dict(sources=sources_config, start=None, end=None, frequency="6h")

    ONE = "1" in sys.argv
    TWO = "2" in sys.argv
    THREE = "3" in sys.argv
    if ONE:
        print("✅-✅")
        test_one(training_context)

    if TWO:
        print("✅--✅")
        test_two(training_context)

    if THREE:
        print("✅---✅")
        test_three(training_context)


if __name__ == "__main__":

    test()
