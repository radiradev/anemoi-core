# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import os
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

    def as_native(self):
        raise NotImplementedError("use as_nested")

    @classmethod
    def new_empty(cls):
        return cls.__class__()

    def as_nested(self):
        tree = {}
        for path, value in self.items():
            node = tree
            for key in path[:-1]:  # walk all but last
                node = node.setdefault(key, {})
            node[path[-1]] = value
        return tree

    @classmethod
    def new_from_native(cls, structure, boxed=True):
        if not isinstance(structure, dict):
            return structure
        if boxed and is_box(structure):
            return structure
        res = cls()
        for prefix, value in structure.items():
            value = cls.new_from_native(value).items()
            if isinstance(value, cls):
                for path, v in cls.new_from_native(value).items():
                    res[(prefix,) + path] = v
            else:
                res[(prefix,)] = value
        return res

    def as_native(self):
        return {k: v.as_native() if isinstance(v, Dict) else v for k, v in self.items()}

    def __repr__(self):
        # this function is quite long and has a lot of knowledge about the other types
        # but it's ok because everything related to display is here
        from rich.tree import Tree

        from anemoi.training.data.refactor.formatting import choose_icon
        from anemoi.training.data.refactor.formatting import format_key_value

        def order_leaf(leaf):
            def priority(k):
                if str(k).startswith("_"):
                    return 10
                return dict(data=1, latitudes=2, longitudes=3, timedeltas=4).get(k, 9)

            order = sorted(leaf.keys(), key=priority)
            assert len(leaf) == len(order), (leaf, leaf.keys())
            return {k: leaf[k] for k in order}

        def expanded_leaf(path, leaf, debug=False):
            if not isinstance(leaf, dict):
                return f"{path}: " + format_key_value(path, leaf)

            assert isinstance(leaf, dict)

            leaf = order_leaf(leaf)
            t = Tree(f"{path}")
            for key, value in leaf.items():
                if not debug and str(key).startswith("_"):
                    continue
                if key == "rollout_usage":
                    subtree = Tree(f"{choose_icon(key, value)} {key} : {value}")
                    t.add(subtree)
                    continue
                if key == "rollout":
                    if debug:
                        subtree = Tree(f"{choose_icon(key, value)} {key} :")
                        for (kind, step), moves in value.rollout_info().items():
                            txt = f"step {step}, {kind}: "
                            txt += ",".join(to_ for from_, _, to_ in moves)
                            txt += " ⟸️  "
                            txt += ",".join(from_ for from_, _, to_ in moves)
                            subtree.add(txt)
                        t.add(subtree)
                    else:
                        t.add(Tree(f"{choose_icon(key, value)} {key} : {value}"))
                    continue
                t.add(choose_icon(key, value) + " " + f"{key} : " + format_key_value(key, value))
            return t

        def debug_leaf(path, leaf):
            return expanded_leaf(path, leaf, debug=True)

        def one_line_leaf(path, leaf):
            leaf = leaf.copy()
            txt = []
            if "data" in leaf:
                x = choose_icon("data", leaf["data"]) + " "
                x += format_key_value("data", leaf.pop("data"))
                x = x.replace("data : ", "")
                x = x[:30] + ("…" if len(x) > 30 else "")
                x += " "
                txt.append(x)
            for k in ["latitudes", "longitudes", "timedeltas"]:
                if k in leaf:
                    txt.append(choose_icon(k, leaf.pop(k)))
            if leaf and txt:
                txt.append(" +")
            for k, v in leaf.items():
                if str(k).startswith("_"):
                    continue
                txt.append(" " + k)

            return Tree(f"{path}: " + "".join(txt))

        name = self.__class__.__name__

        for leaf in self.values():
            if isinstance(leaf, dict) and "_reference_date" in leaf:
                name += f" (Reference {leaf['_reference_date']})"
                break

        tree = Tree(name)
        verbose = int(os.environ.get("ANEMOI_CONFIG_VERBOSE_STRUCTURE", 0))
        leaf_to_tree = {0: one_line_leaf, 1: expanded_leaf, 2: debug_leaf}[verbose]
        for path, leaf in self.items():
            path = ".".join(str(p) for p in path)
            if not isinstance(leaf, dict):
                tree.add(f"{path}: " + format_key_value(path, leaf))
                continue
            if isinstance(leaf, dict) and not leaf:
                tree.add(f"{path}: ❌ <empty-dict>")
                continue
            assert isinstance(leaf, dict)
            tree.add(leaf_to_tree(path, leaf))

        console = Console(record=True)
        with console.capture() as capture:
            console.print(tree, overflow="ellipsis")
        return capture.get()

    def to_str(self, name):
        return name + " " + self.__repr__()

    @classmethod
    def _split_path(self, path):
        if isinstance(path, str):
            path = tuple(int(x) if x.isdigit() else x for x in path.split("."))
        assert isinstance(path, tuple), type(path)
        return path

    def __setitem__(self, path, value):
        path = self._split_path(path)
        if isinstance(value, self.__class__):
            for p, v in value.items():
                self[path + p] = v
            return
        super().__setitem__(path, value)

    def __getitem__(self, path):
        path = self._split_path(path)
        matching = {p[len(path) :]: value for p, value in self.items() if p[: len(path)] == path}
        if not matching:
            raise KeyError(f"Path {path} not found in Dict. Available paths: {list(self.keys())}")
        if () in matching:  # exact match
            return matching[()]
        return Dict(matching)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'Dict' object has no attribute '{name}'")

    def as_function(self):
        return self._cast(Function, check_leaf_type=lambda x: callable(x))

    def as_module_dict(self):
        return self._cast(AnemoiModuleDict, check_leaf_type=lambda x: isinstance(x, torch.nn.Module))

    def as_batch(self):
        return self._cast(Batch)

    def _cast(self, cls, check_leaf_type=lambda x: True):
        res = cls()
        for path, v in self.items():
            if not check_leaf_type(v):
                raise ValueError(f"Unexpected leaf at {path}, got {type(v)}")
            res[path] = v
        return res

    def wrap_in_box(self, key):
        return self.leaf_accessor.map(lambda v: {key: v})

    def select_content(self, *keys):
        """Usage:
        select_content(["latitudes", "longitudes"]) returns a Dict with boxes containing only the selected keys
        select_content("data") returns a Dict with the content of the "data" key, no box
        select_content("latitudes", "longitudes") returns a tuple of Dicts, each containing the content of the corresponding key, no boxing
        select_content(["latitudes"], ["longitudes"]) returns a tuple of Dicts, each containing the content of the corresponding key, with boxing
        """
        if len(keys) == 0:
            raise ValueError("At least one key must be provided")
        if len(keys) > 1:
            return tuple(self.select_content(k) for k in keys)

        assert len(keys) == 1
        keys = keys[0]

        def select(box):
            if isinstance(keys, (list, tuple)):
                return {k: v for k, v in box.items() if k in keys}
            _check_key_in_dict(box, keys)
            return box[keys]

        return self.each.map(select)

    def merge_leaves(self, *args, **kwargs):
        new = self.copy()
        new.each.update(*args, **kwargs)
        return new

    @property
    def each(self):
        return LeafAccessor(self)


class LeafAccessor:
    def __init__(self, parent):
        self.parent = parent

    def copy(self):
        return self.parent.__class__(self._apply_dict_method("copy"))

    def pop(self, *args, **kwargs):
        return self.parent.__class__(self._apply_dict_method("pop", *args, **kwargs))

    def popitem(self, *args, **kwargs):
        return self.parent.__class__(self._apply_dict_method("popitem", *args, **kwargs))

    def reversed(self):
        print("untested code")
        return self.parent.__class__(self._apply_dict_method("reversed"))

    def values(self):
        print("untested code")
        return self.parent.__class__(self._apply_dict_method("values"))

    def __or__(self, other):
        print("untested code")
        return self.parent.__class__(self._apply_dict_method("__or__", other))

    def __getitem__(self, key: str):
        return self.parent.__class__(self._apply_dict_method("__getitem__", key))

    def __setitem__(self, key: str, value):
        self._apply_dict_method("__setitem__", key, value)

    def update(self, *args, **kwargs):
        self._apply_dict_method("update", *args, **kwargs)

    def __ior__(self, other):
        print("untested code")
        self._apply_dict_method("__ior__", other)

    def _apply_dict_method(self, method_name, *args, **kwargs):
        def function_finder(path, leaf):
            if not isinstance(leaf, dict):
                raise ValueError(f"Expected dict at {path}, got {type(leaf)}")
            return getattr(leaf, method_name)

        return self._parallel_apply_on_leaves(function_finder, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        def function_finder(path, leaf):
            if not callable(leaf):
                raise ValueError(f"Expected callable at {path}, got {type(leaf)}")
            return leaf

        res = self.parent.__class__()
        for k, v in self._parallel_apply_on_leaves(function_finder, *args, **kwargs):
            res[k] = v
        return res

    def map(self, func, *args, **kwargs):
        res = self.parent.__class__()
        for path, leaf in self.parent.items():
            args_ = [a[path] if isinstance(a, Dict) else a for a in args]
            kwargs_ = {k: v[path] if isinstance(v, Dict) else v for k, v in kwargs.items()}
            new = func(leaf, *args_, **kwargs_)
            res[path] = new
        return res

    def _parallel_apply_on_leaves(self, func_finder, *args, **kwargs):
        output = []
        for path, leaf in self.parent.items():
            # when args or kwargs are Dict, extract the corresponding leaf
            args_ = [a[path] if isinstance(a, Dict) else a for a in args]
            kwargs_ = {k: v[path] if isinstance(v, Dict) else v for k, v in kwargs.items()}
            # find the function/method to apply and apply it
            func = func_finder(path, leaf)
            res = func(*args_, **kwargs_)
            # stack in a list so that the caller decides how to handle it
            output.append((path, res))
        return output


class Batch(Dict):
    pass


class AnemoiModuleDict(torch.nn.ModuleDict):
    emoji = "🔥"

    def __init__(self, *args, **kwargs):
        dic = dict(*args, **kwargs)

        def deep_cast(x, cls):
            if isinstance(x, dict):
                return cls({k: deep_cast(v, cls) for k, v in x.items()})
            if not isinstance(x, torch.nn.Module):
                raise ValueError(f"Expected nn.Module, got {type(x)}")
            return x

        dic = deep_cast(dic, AnemoiModuleDict)
        self.__init__(dic)

    def __call__(self, *args, **kwargs):
        first = None
        if args:
            first = args[0]
            res = first.__class__()
        elif kwargs:
            first = next(iter(kwargs.values()))
            res = first.__class__()
        else:
            res = Dict()

        for path, module in self.items():
            print(f"applying module {module} at path {path}")
            args_ = [a[path] if isinstance(a, Dict) else a for a in args]
            kwargs_ = {k: v[path] if isinstance(v, Dict) else v for k, v in kwargs.items()}
            res[path] = module(*args_, **kwargs_)
        return res


class Function(Dict):
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

            return key, res

        res = _remap(self, visit=apply)
        return Dict(res)

    def apply_on_multiple_structures(self, *args, _output_box=True, **kwargs):

        def apply(path, key, func):
            if not callable(func):
                return key, func

            def search(x):
                return _get_path(x, path + (key,), default=None)

            args = [search(a) if isinstance(a, Dict) else a for a in args]
            kwargs = {k: search(v) if isinstance(v, Dict) else v for k, v in kwargs.items()}

            res = func(*args, **kwargs)
            if _output_box:
                if not isinstance(res, dict):
                    raise ValueError(f"Function at {path}, {key} did not return a dict, got {type(res)}")
            return res

        res = _remap(self, visit=apply)
        return Dict(res)

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
    return not isinstance(structure, (dict, list, tuple, torch.nn.ModuleDict))


def is_box(structure):
    if not isinstance(structure, dict):
        assert False, structure
        return False
    # See inside the dict if there is a final element
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
        res = {}
        for struct in structs:
            box = _get_path(struct, path + (key,), default={})
            if not overwrite and set(box.keys()) & set(res.keys()):
                raise ValueError(f"Conflicting keys found in structures: {set(box.keys()) & set(res.keys())}")
            res.update(**box)
        return res

    return _remap(structs[0], exit=exit)


def _stop_if_box_enter(path, key, value):
    if is_box(value):
        return value, []
    return _default_enter(path, key, value)


def ensure_output_is_box_exit(path, key, old_parent, new_parent, new_items):
    lkj
    return res


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
    return Dict(root)


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


            """

    print("✅✅✅✅✅✅✅✅✅✅✅✅✅✅")
    config = yaml.safe_load(cfg_1)

    sp = sample_provider_factory(**training_context, **config)
    print(sp)
    schema = sp.dataschema

    # print(schema)
    # print(repr_schema(schema, "schema"))

    print("✅ sp.static_info", sp.static_info)

    batch_data = sp[1]
    for k, v in batch_data.items():
        print(f" - {k}: {type(v)}")
    print("✅ Batch Data", batch_data)

    data = sp.static_info.copy()
    data.each.update(batch_data)
    print("✅ Data full", data)
    from anemoi.training.data.refactor.structure import Dict

    assert isinstance(data, Dict), type(data)
    assert "data" in data["lowres"], data["lowres"].keys()
    assert "name_to_index" in data["lowres"], data["lowres"].keys()

    def to_tensor(box):
        return {k: torch.Tensor(v) if k == "data" else v for k, v in box.items()}

    def to_gpu(box):
        return {k: v.to("cuda") if k == "data" else v for k, v in box.items()}

    new_data = data.each.map(to_tensor)
    new_data = new_data.each.map(to_gpu)
    print("✅ Data on GPU", new_data)
    print(type(new_data["lowres"]))

    # guessed = make_schema(data)
    # print(to_str(schema, "Actual Schema"))
    # print(to_str(guessed, "Actual Schema"))

    extra = data.each.pop("extra")
    print("Extra after pop extra", extra)

    # lat,lon = data.select_content("latitudes", "longitudes")
    lat = data.each["latitudes"]
    lon = data.each["longitudes"]
    print("Latitudes", lat)
    print("Longitudes", lon)

    def build_normaliser(box):
        statistics = box["statistics"]
        mean = np.mean(statistics["mean"])

        def func(box):
            box = box.copy()
            box["data"] = box["data"] - mean
            return box

        return func

    # mimic this:
    # normaliser = {}
    # for path, value in sp.static_info.items():
    #     normaliser[path] = build_normaliser(value)

    # normaliser = Dict()
    # normaliser = sp.static_info.new_empty()
    # for path, value in sp.static_info.items():
    #     normaliser[path] = build_normaliser(value)

    normaliser = sp.static_info.each.map(build_normaliser)

    print("Normaliser function", normaliser)
    n_data = normaliser.each(data)

    d = data.each["data"]
    n = n_data.each["data"]
    print_columns(f"Unnormalised data {d}", f"Normalised data {n}")


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
                state:
                   dictionary:
                     fields:
                       rollout: prognostics
                       # rollout: forcings
                       # rollout: diagnostics
                       container:
                           dimensions: ["values", "variables"]
                           variables: ["2t", "10u", "10v"]
                           data_group: "era5"

                  #observations:
                  #  tuple:
                  #    for_each:
                  #      - offset: ["-6h", "0h", "+6h"]
                  #    template:
                  #      tensor:
                  #        - variables: ["metop_a.scatss_1", "metop_a.scatss_2"]
            """

    rollout_config = yaml.safe_load(
        """
        rollout:
          steps: ["0h", "+6h", "+12h", "+18h", "+24h"]
          input: ["-12h", "-6h", "0h"]
          target: ["+6h", "12h"]
        """,
    )
    print("Rollout config", rollout_config)

    config = yaml.safe_load(cfg_2)
    sp = sample_provider_factory(**training_context, **config, **rollout_config)
    print("static_info", sp.static_info)
    data = sp[1]
    print(data)
    data = sp.static_info.merge_leaves(data)
    data = data.select_content(["_offset", "data", "_reference_date"])
    # data = data.select_content(["_offset", "rollout_usage", "data"])
    print("Selected Data", data)
    print("Rollout info : ", type(sp.rollout_info()), sp.rollout_info())
    # merged = sp.rollout_info().merge_leaves(data)
    # print("merged", merged)
    rollout_steps = None
    for path, rollout in sp.rollout_info().items():
        rollout_steps = rollout.steps
        break
    print("Rollout steps", rollout_steps)

    rollout = sp.rollout_info()
    print("Rollout info", rollout)
    # step 0:

    data.each["_tag"] = "✅ data"
    input = None
    output = None
    for i, step in enumerate(rollout_steps):
        input = rollout.each("input", step=step, database=data, previous_input=input, previous_output=output)
        target = rollout.each("target", step=step, database=data)
        # run model
        output = target
        print(f"-------- Rollout {i} {step=} --------")
        print_columns(input.to_str(f"Input at step {i}"), target.to_str(f"Target at step {i}"))
        output.each["_tag"] = "💬 previous_output"
        input.each["_tag"] = "😊  previous_input"


def test_three(training_context):
    from anemoi.training.data.refactor.sample_provider import sample_provider_factory

    cfg_3 = """dictionary:
                  ams:
                    for_each:
                      - offset: ["-6h", "0h"]
                      - container:
                          variables: ["scatss_1", "scatss_2"]
                          data_group: "metop_a"
            """
    config = yaml.safe_load(cfg_3)
    sp = sample_provider_factory(**training_context, **config)
    print(sp)
    s = sp.static_info
    print(s)
    for path, box in s.boxes():
        print(box.to_str(f"Box at {path}"))
    print("--------------")
    print(s["ams", "-6h"])
    s["ams", "0h"] = s["ams", "-6h"]
    print(s["ams.0h"]["_offset"])

    nested = sp.static_info

    modified = nested.empty_like()
    for path, box in nested.boxes():
        box["new_key"] = 48
        modified[path] = box
    print(modified.to_str("Modified static info"))


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
