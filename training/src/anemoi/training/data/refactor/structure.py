# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import datetime
import json
import os
from functools import wraps

import numpy as np
import torch
import yaml
from rich import print
from rich.console import Console
from rich.tree import Tree

from anemoi.utils.dates import frequency_to_string


def format_shape(k, v):
    return f"shape: {v}"


def format_dict(k, v):
    if not isinstance(v, dict):
        return "❌" + format_default(k, v)
    keys = ",".join(f"{k_}" for k_ in v.keys())
    return format_shorten(k, "dict with keys " + keys)


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


def format_default(k, v):
    if isinstance(v, (int, float, bool)):
        return f"{k} : {v}"
    if isinstance(v, str):
        return f"{k} : '{v}'"
    if isinstance(v, (list, tuple)):
        return format_shorten(k, str(v))
    if isinstance(v, np.ndarray):
        return f"{k} : np.array of shape {v.shape} with mean {np.nanmean(v):.2f}"
    if isinstance(v, dict):
        return format_dict(k, v)
    try:
        import torch

        if isinstance(v, torch.Tensor):
            shape = ", ".join(str(dim) for dim in v.size())
            return f"{k} : tensor of shape ({shape}) on {v.device}"
    except ImportError:
        pass
    return f"{k} : {v.__class__.__name__}"


def format_none(k, v):
    return None


def _verbose_structure():
    # TODO: should read from utils.configs
    return os.environ.get("ANEMOI_CONFIG_VERBOSE_STRUCTURE") != "0"


def format_key_value(key, v):
    return format_default(key, v)


def repr_schema(schema, name="schema"):
    tree = _repr_schema(schema, name=name)
    return _tree_to_string(tree)


def _repr_schema(schema, name="schema"):
    if "type" not in schema:
        raise ValueError(f"schema must contain 'type' key: {schema}")
    _type = schema["type"]

    if _type in ["dict", "box"]:
        if "children" not in schema:
            raise ValueError(f"Dict/tuple/box schema must contain 'children' key: {schema}")
        if not isinstance(schema["children"], dict):
            raise ValueError(f"Inconsistent schema, type={_type} but 'children' is not a dict: {schema}")
    if _type in ["tuple"]:
        if "children" not in schema:
            raise ValueError(f"Dict/tuple/box schema must contain 'children' key: {schema}")
        if not isinstance(schema["children"], (list, tuple)):
            raise ValueError(f"Inconsistent schema, type={_type} but 'children' is not a list or tuple: {schema}")

    if _type == "dict":
        tree = Tree(f"📖 {name}: {_type}")
        for k, v in schema["children"].items():
            tree.add(_repr_schema(v, name=k))
    elif _type == "box":
        tree = Tree(f"📦 {name}: {_type}")
        for k, v in schema["children"].items():
            tree.add(_repr_schema(v, name=k))
    elif _type == "tuple":
        tree = Tree(f"🔗 {name}: {_type}")
        for i, v in enumerate(schema["children"]):
            tree.add(_repr_schema(v, name=str(i)))
    else:
        tree = Tree(f"🌱 {name}: {_type}")
    return tree


def repr_function(func, name="Function"):
    if not callable(func):
        raise ValueError("func is not callable")
    if not hasattr(func, "_anemoi"):
        raise ValueError("func does not have '_anemoi' attribute")
    return _repr(func._anemoi["structure"], name, _boxed=False)


def repr(nested, name="Data", **kwargs):
    return repr_box(nested, name, **kwargs)


def repr_leaf(nested, name="Nested", **kwargs):
    return _repr(nested, name, _boxed=False, **kwargs)


def repr_box(nested, name="Batch", **kwargs):
    return _repr(nested, name, _boxed=True, **kwargs)


def _repr(nested, name, _boxed=True, _schema=None):
    if _schema is None:
        _schema = guess_schema(nested)
    tree = _tree(_schema, name, nested, _boxed=_boxed)
    return _tree_to_string(tree)


def _tree_to_string(tree):
    console = Console(record=True, width=120)
    with console.capture() as capture:
        console.print(tree)
    return capture.get()


def _tree(schema, key, nested, **kwargs):
    box = "📦 " if kwargs.get("_boxed") else ""
    leaf = "🌱 "
    # leaf = "" if _boxed else "🌱 "
    if final_schema(schema):
        txt = format_key_value(key, nested)
        return Tree(f"{leaf}{txt}")

    if box_schema(schema):
        if not _verbose_structure():
            return Tree(f"{box}{key} : {" ".join(f"{k}" for k in nested)}")

        tree = Tree(f"{box}{key} :")
        for k, v in nested.items():
            txt = format_key_value(k, v)
            if txt is not None:
                tree.add(f"{leaf}{txt}")
        return tree

    if tuple_schema(schema):
        tree = Tree(str(key))
        for i, v in enumerate(nested):
            tree.add(_tree(schema["children"][i], i, v, **kwargs))
        return tree

    if dict_schema(schema):
        tree = Tree(str(key))
        for k, v in nested.items():
            tree.add(_tree(schema["children"][k], k, v, **kwargs))
        return tree

    raise ValueError(f"Unknown schema type: {schema}")


def _all_arguments(args, kwargs):
    for i, v in enumerate(args):
        yield (i, v)
    for k, v in kwargs.items():
        yield (k, v)


def _recursable_arguments(args, kwargs, no_recurse):
    for k, v in _all_arguments(args, kwargs):
        if k in no_recurse:
            continue
        yield (k, v)


def probe_value(value):
    if isinstance(value, dict):
        assert isinstance(value, dict)
        for k, v in value.items():
            p = probe_value(v)
            if final_schema(p):
                return dict(type="box")
        return dict(type="dict")
    if isinstance(value, (list, tuple)):
        return dict(type="tuple")
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return dict(type="tensor")
    if callable(value):
        return dict(type="callable")
    return dict(type=type(value).__name__)


def check_schema(schema):
    if not isinstance(schema, dict):
        raise ValueError(f"Expected schema to be a dict, got {type(schema)} : {schema}")
    assert isinstance(schema.get("type"), str), f"schema must contain a 'type' key with a string value: {schema}"


def final_schema(schema):
    return schema["type"] not in [
        "box",
        "dict",
        "tuple",
    ]


def box_schema(schema):
    return schema["type"] == "box"


def dict_schema(schema):
    return schema["type"] == "dict"


def tuple_schema(schema):
    return schema["type"] == "tuple"


def guess_schema(*args):
    return _guess_schema(*args)


def _guess_schema(*args, kwargs={}, no_recurse=[]):
    # force kwargs and no_recurse to be keywords to avoid name collision
    kwargs = dict(_recursable_arguments(args, kwargs, no_recurse))

    # if any arguments is final, we found the schema
    for k, v in kwargs.items():
        schema = probe_value(v)
        if final_schema(schema):
            return schema

    # else, all arguments in kwargs are not final
    # return the schema of the first recursable argument
    first = kwargs[next(iter(kwargs))]
    schema = probe_value(first)
    if tuple_schema(schema):
        # found a tuple or a list, add sub-schema
        schema["children"] = [_guess_schema(v_) for v_ in first]
    elif dict_schema(schema) or box_schema(schema):
        # found a dict or a box, add sub-schema
        schema["children"] = {k_: _guess_schema(v_) for k_, v_ in first.items()}
        if box_schema(schema):
            for k, v in schema["children"].items():
                if final_schema(v):
                    continue
                # This is a little hacky here, so let's add an assert
                # the schema are infered deeply in schema['children']
                # but we want(?) to stop at the box level.
                assert len(v) == 2, f"Expected 2 elements in schema for {k}, got {len(v)}: {v}"
                v["type"] = "dict" if v["type"] == "box" else v["type"]
                v["children"] = {}

            if any(box_schema(v) for k, v in schema["children"].items()):
                schema["type"] = "dict"
            else:
                schema["type"] = "box"
    return schema


def assert_compatible_schema(a, b):
    log = [
        (dict_schema(a), dict_schema(b)),
        (box_schema(a), box_schema(b)),
        (tuple_schema(a), tuple_schema(b)),
        (final_schema(a), final_schema(b)),
    ]
    if tuple_schema(a):
        assert tuple_schema(b), f"Schema mismatch: {a} vs {b}"
        a = a["children"]
        b = b["children"]
        assert len(a) == len(b), f"schema mismatch: {len(a)} != {len(b)} in {a} vs {b}"
        for a_, b_ in zip(a, b):
            assert_compatible_schema(a_, b_)
    elif dict_schema(a):
        assert dict_schema(b), f"schema mismatch: {a} vs {b}"
    elif box_schema(a):
        assert box_schema(b), f"schema mismatch: {a} vs {b}"
    elif tuple_schema(a) or box_schema(a):
        a = a["children"]
        b = b["children"]
        assert len(a) == len(b), f"schema mismatch: {len(a)} != {len(b)} in {a} vs {b}"
        for k in a:
            assert_compatible_schema(a[k], b[k])
    else:
        assert final_schema(a) and final_schema(b), f"schema mismatch: {a} vs {b}. {log}"


def compare_schema(a, b):
    if isinstance(a, list):
        a = tuple(a)
    if isinstance(b, list):
        b = tuple(b)
    if isinstance(a, tuple):
        return all(compare_schema(a_, b_) for a_, b_ in zip(a, b))
    if isinstance(a, dict):
        if (final_schema(a) or box_schema(a)) and (final_schema(b) or box_schema(b)):
            return True
        if set(a.keys()) != set(b.keys()):
            return False
        for k in set(a.keys()) | set(b.keys()):
            if k not in b:
                return False
            if not compare_schema(a[k], b[k]):
                return False
        return True
    return a == b


def _call_func_now_and_make_callable_if_needed(*args, _make_callable=False, **kwargs):
    structure = _call_func_now_not_callable(*args, **kwargs)
    if not _make_callable:
        return structure
    print(f"----------- *args ={args}, kwargs={kwargs}")
    print(repr_leaf(structure, name="Structure"))
    schema = kwargs.get("schema")
    func = _structure_to_callable(structure, schema=schema)
    print(repr_function(func, name="Function"))
    return func


def _call_func_now_not_callable(func, args, kwargs, _apply_on_boxes, no_recurse=[], schema=None):
    # print(f"DEBUG: Applying {func.__name__} with args={args}, kwargs={kwargs}, no_recurse={no_recurse}, schema={schema}")
    if schema is None:
        schema = _guess_schema(*args, kwargs=kwargs, no_recurse=no_recurse)
    check_schema(schema)
    # print("DEBUG", repr_schema(schema))

    if final_schema(schema):
        return func(*args, **kwargs)

    if _apply_on_boxes and box_schema(schema):
        return func(*args, **kwargs)

    def recurse(v, name, key):
        # pick key in dict only if argument is recursable
        return v[key] if name not in no_recurse else v

    def next_apply(key):
        return _call_func_now_not_callable(
            func,
            args=[recurse(v, i, key) for i, v in enumerate(args)],
            kwargs={k_: recurse(v, k_, key) for k_, v in kwargs.items()},
            no_recurse=no_recurse,
            _apply_on_boxes=_apply_on_boxes,
            schema=schema["children"][key],
        )

    if dict_schema(schema) or (box_schema(schema) and not _apply_on_boxes):
        keys = schema["children"].keys()
        for k, v in _recursable_arguments(args, kwargs, no_recurse):
            if not isinstance(v, dict):
                raise ValueError(f"Expected dict content for argument {k}, got {type(v)}. {schema}")
            if set(v.keys()) != set(keys):
                raise ValueError(f"Keys mismatch for {k}: {set(v.keys())} != {set(keys)}, {_apply_on_boxes}")
        return {key: next_apply(key) for key in keys}

    if tuple_schema(schema):
        length = len(schema["children"])
        for k, v in _recursable_arguments(args, kwargs, no_recurse):
            if not isinstance(v, (list, tuple)):
                raise ValueError(f"Expected list or tuple content for argument {k}, got {type(v)}. {schema}")
            if len(v) != length:
                raise ValueError(f"Length mismatch for {k}: {len(v)} != {length} in {schema}")
        return tuple([next_apply(key) for key in range(length)])
    raise ValueError(f"Unknown schema type: {schema}")


# def function_on_box(*args=None,**options):
#    return apply_to_each_box(callable_or_schema, **options)
#
# def function_on_leaf(callable_or_schema=None,**options):
#    raise NotImplementedError("TODO")


def _structure_to_callable(structure, schema=None):
    """From a structure with all leaves callable, create a function to be apply to strctures with same schema"""
    if schema is None:
        schema = guess_schema(structure)

    @apply_to_each_leaf
    def assert_is_callable(x):
        assert callable(x), f"Expected callable, got {type(x)}, in {repr(structure)}"

    assert_is_callable(structure)

    def func(*args, **kwargs):

        @apply_to_each_leaf
        def func_(x, *args, **kwargs):
            return x(*args, **kwargs)

        return func_(structure, *args, **kwargs)

    func._anemoi = dict(structure=structure, schema=schema)
    return func


def function_on_box(callable_or_schema=None, **options):
    return _apply_on_x(callable_or_schema, **options, _apply_on_boxes=True, _make_callable=True)


def function_on_leaf(callable_or_schema=None, **options):
    return _apply_on_x(callable_or_schema, **options, _apply_on_boxes=False, _make_callable=True)


def apply_to_each_leaf(callable_or_schema=None, **options):
    return _apply_on_x(callable_or_schema, **options, _apply_on_boxes=False, _make_callable=False)


def apply_to_each_box(callable_or_schema=None, **options):
    return _apply_on_x(callable_or_schema, **options, _apply_on_boxes=True, _make_callable=False)


def _apply_on_x(callable_or_schema=None, **options):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return _call_func_now_and_make_callable_if_needed(func, args, kwargs, **options)

        return wrapper

    # decorator called with no parameters, apply the wrapper around the first arg
    no_arguments = set(options.keys()) == {"_apply_on_boxes", "_make_callable"}
    if no_arguments and callable(callable_or_schema):
        return inner(callable_or_schema)

    # decorator called with parameters, if present, first arg is the schema
    schema = callable_or_schema
    if options.get("schema") is not None and schema is not None:
        raise ValueError("Multiple values for argument 'schema'.")
    options["schema"] = schema

    return inner


@apply_to_each_box
def merge(a, b):
    if not isinstance(a, dict):
        raise ValueError(f"Expected dict for a, got {type(a)}")
    if not isinstance(b, dict):
        raise ValueError(f"Expected dict for b, got {type(b)}")
    return {**a, **b}


@apply_to_each_box
def _pop(a, key):
    if not isinstance(a, dict):
        raise ValueError(f"Expected dict for a, got {type(a)}")
    if key not in a:
        raise ValueError(f"Key {key} not found in dict. Available keys are: {list(a.keys())}")
    value = a.pop(key)
    return {key: value}


@apply_to_each_box(no_recurse=["key"])
def _pop(a, key):
    if not isinstance(a, dict):
        raise ValueError(f"Expected dict for a, got {type(a)}")
    if key not in a:
        raise ValueError(f"Key {key} not found in dict. Available keys are: {list(a.keys())}")
    value = a.pop(key)
    return {key: value}


def pop(a, key):
    return _pop(a, key=key)


@apply_to_each_box(no_recurse=["key", "default"])
def _filter(a, key):
    if not isinstance(a, dict):
        raise ValueError(f"Expected dict for a, got {type(a)}")
    if key not in a:
        raise ValueError(f"Key {key} not found in dict. Available keys are: {list(a.keys())}")
    return {key: a.get(key)}


def filter(a, key):
    return _filter(a, key=key)


class Structure:
    def __init__(self, content, schema=None):
        self.content = content
        self.schema = schema

    def __call__(self, *args, **kwargs):
        @apply_to_each_leaf()
        def wrapper(content, *_args, **_kwargs):
            return content(*_args, **_kwargs)

        return wrapper(self.content, *args, **kwargs)


def as_structure(content):
    return Structure(content)


def test(ONE=True, TWO=True, THREE=True):
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

    cfg = """dictionary:
                input:
                  dictionary:
                    fields:
                    #  tensor:
                         variables: ["era5.2t", "era5.10u", "era5.10v"]
                    #    - offset: ["-6h"]
                    #other_fields:
                    #  tuple:
                    #    loop:
                    #      - offset: ["-6h", "+6h"]
                    #    template:
                    #      variables: ["era5.2t", "era5.10u"]
                    #observations:
                    #  tuple:
                    #    loop:
                    #      - offset: ["-6h", "0h", "+6h"]
                    #    template:
                    #      tensor:
                    #        - variables: ["metop_a.scatss_1", "metop_a.scatss_2"]
        """
    config = yaml.safe_load(cfg)

    print("✅✅✅✅✅✅✅✅✅✅✅✅✅✅")
    from anemoi.training.data.refactor.sample_provider import sample_provider_factory

    sp = sample_provider_factory(**training_context, **config)
    schema = sp.dataschema

    # print(schema)
    print(repr_schema(schema, "schema"))

    print(repr(sp.static_info, name="✅ sp.static_info"))
    print(sp.static_info)

    data = sp[1]
    print(repr(data, name="✅ Data"))
    print(data)

    guessed = guess_schema(data)
    # print(f"Guessed schema: {repr_schema(guessed)}")
    # print("schema=", repr_schema(schema))
    assert_compatible_schema(schema, guessed)

    data = merge(data, sp.static_info)
    print(repr(data, name="✅ Data merged with sp.static_info"))
    print(guess_schema(data))

    pop(data, "extra")
    print(repr(data, name="Data after popping extra"))

    print("----- select -------")

    latitudes = filter(data, "latitudes")
    print(repr(latitudes, name="Latitudes"))

    exit()

    longitudes = filter(data, key="longitudes")
    print(repr(longitudes, name="Long"))
    print(guess_schema(longitudes))

    print("------")

    @function_on_box
    def build_normaliser(statistics):
        mean = np.mean(statistics["longitudes"])
        # mean = statistics["mean"]

        def func(box, device):
            box[data].to_device(device)
            box["normalized_data"] = box["data"] - mean
            return box
            # return {apply_on_key: box[apply_on_key] - mean}

        # norm = Normaliser(mean,...)
        # def f(x):
        #    return norm.transform(x["data"])

        return func

    normaliser = build_normaliser(data)
    print(repr(data))
    print(repr(normaliser(data, device="gpu"), name="normaliser(data)"))
    # print(repr(normaliser(data, apply_on_key="data"), name="normaliser(data)"))

    exit()

    @apply_to_each_leaf
    def apply_func(func, data):
        return func(data)

    print(
        repr(
            apply_func(normalisers, data=data),
            name="apply_func(normalisers, data=data)",
        ),
    )
    print(sp.static_info)
    print(guess_schema(sp.static_info))
    print(repr(sp.static_info, name="sp.static_info"))


if __name__ == "__main__":
    test(False, False, True)
