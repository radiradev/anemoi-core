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


def repr_specs(specs, name="Specs"):
    tree = _repr_specs(specs, name=name)
    return _tree_to_string(tree)


def _repr_specs(specs, name="Specs"):
    if "type" not in specs:
        raise ValueError(f"Specs must contain 'type' key: {specs}")
    _type = specs["type"]

    if _type in ["dict", "box", "tuple"]:
        if "specs" not in specs:
            raise ValueError(f"Dict specs must contain 'specs' key: {specs}")

    if _type in ["dict", "box"]:
        if not isinstance(specs["specs"], dict):
            raise ValueError(f"Inconsistent specs, type={_type} but specs is not a dict: {specs}")
    if _type in ["tuple"]:
        if not isinstance(specs["specs"], (list, tuple)):
            raise ValueError(f"Inconsistent specs, type={_type} but specs is not a list or tuple: {specs}")

    if _type == "dict":
        tree = Tree(f"📖 {name}: {_type}")
        for k, v in specs["specs"].items():
            tree.add(_repr_specs(v, name=k))
    elif _type == "box":
        tree = Tree(f"📦 {name}: {_type}")
        for k, v in specs["specs"].items():
            tree.add(_repr_specs(v, name=k))
    elif _type == "tuple":
        tree = Tree(f"🔗 {name}: {_type}")
        for i, v in enumerate(specs["specs"]):
            tree.add(_repr_specs(v, name=str(i)))
    else:
        tree = Tree(f"🌱 {name}: {_type}")
    return tree


def repr_function(func, name="Function"):
    if not callable(func):
        raise ValueError("func is not callable")
    if not hasattr(func, "_anemoi"):
        raise ValueError("func does not have '_anemoi' attribute")
    return _repr(func._anemoi["structure"], name, _boxed=False)


def repr(nested, name="Nested", **kwargs):
    return repr_leaf(nested, name, **kwargs)


def repr_leaf(nested, name="Nested", **kwargs):
    return _repr(nested, name, _boxed=False, **kwargs)


def repr_box(nested, name="Batch", **kwargs):
    return _repr(nested, name, _boxed=True, **kwargs)


def _repr(nested, name, _boxed=True, _specs=None):
    if _specs is None:
        _specs = guess_specs(nested)
    tree = _tree(_specs, name, nested, _boxed=_boxed)
    return _tree_to_string(tree)


def _tree_to_string(tree):
    console = Console(record=True, width=120)
    with console.capture() as capture:
        console.print(tree)
    return capture.get()


def _tree(specs, key, nested, **kwargs):
    box = "📦 " if kwargs.get("_boxed") else ""
    leaf = "🌱 "
    # leaf = "" if _boxed else "🌱 "
    if specs_is_final(specs):
        txt = format_key_value(key, nested)
        return Tree(f"{leaf}{txt}")

    if specs_is_box(specs):
        if not _verbose_structure():
            return Tree(f"{box}{key} : {" ".join(f"{k}" for k in nested)}")

        tree = Tree(f"{box}{key} :")
        for k, v in nested.items():
            txt = format_key_value(k, v)
            if txt is not None:
                tree.add(f"{leaf}{txt}")
        return tree

    if specs_is_tuple(specs):
        tree = Tree(str(key))
        for i, v in enumerate(nested):
            tree.add(_tree(specs["specs"][i], i, v, **kwargs))
        return tree

    if specs_is_dict(specs):
        tree = Tree(str(key))
        for k, v in nested.items():
            tree.add(_tree(specs["specs"][k], k, v, **kwargs))
        return tree

    raise ValueError(f"Unknown specs type: {specs}")


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
            if specs_is_final(p):
                return dict(type="box")
        return dict(type="dict")
    if isinstance(value, (list, tuple)):
        return dict(type="tuple")
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return dict(type="tensor")
    if callable(value):
        return dict(type="callable")
    return dict(type=type(value).__name__)


def check_specs(specs):
    if not isinstance(specs, dict):
        raise ValueError(f"Expected specs to be a dict, got {type(specs)} : {specs}")
    assert isinstance(specs.get("type"), str), f"Specs must contain a 'type' key with a string value: {specs}"


def specs_is_final(spec):
    return spec["type"] not in [
        "box",
        "dict",
        "tuple",
    ]


def specs_is_box(specs):
    return specs["type"] == "box"


def specs_is_dict(specs):
    return specs["type"] == "dict"


def specs_is_tuple(specs):
    return specs["type"] == "tuple"


def guess_specs(*args):
    return _guess_specs(*args)


def _guess_specs(*args, kwargs={}, no_recurse=[]):
    # force kwargs and no_recurse to be keywords to avoid name collision
    kwargs = dict(_recursable_arguments(args, kwargs, no_recurse))

    # if any arguments is final, we found the specs
    for k, v in kwargs.items():
        spec = probe_value(v)
        if specs_is_final(spec):
            return spec

    # else, all arguments in kwargs are not final
    # return the specs of the first recursable argument
    first = kwargs[next(iter(kwargs))]
    spec = probe_value(first)
    if specs_is_tuple(spec):
        # found a tuple or a list, add sub-specs
        spec["specs"] = [_guess_specs(v_) for v_ in first]
    elif specs_is_dict(spec) or specs_is_box(spec):
        # found a dict or a box, add sub-specs
        spec["specs"] = {k_: _guess_specs(v_) for k_, v_ in first.items()}
        if specs_is_box(spec):
            for k, v in spec["specs"].items():
                if specs_is_final(v):
                    continue
                # This is a little hacky here, so let's add an assert
                # the specs are infered deeply in spec['specs']
                # but we want(?) to stop at the box level.
                assert len(v) == 2, f"Expected 2 elements in spec for {k}, got {len(v)}: {v}"
                v["type"] = "dict" if v["type"] == "box" else v["type"]
                v.pop("specs")

            if any(specs_is_box(v) for k, v in spec["specs"].items()):
                spec["type"] = "dict"
            else:
                spec["type"] = "box"
    return spec


def assert_compatible_specs(a, b):
    log = [
        (specs_is_dict(a), specs_is_dict(b)),
        (specs_is_box(a), specs_is_box(b)),
        (specs_is_tuple(a), specs_is_tuple(b)),
        (specs_is_final(a), specs_is_final(b)),
    ]
    if specs_is_tuple(a):
        assert specs_is_tuple(b), f"Specs mismatch: {a} vs {b}"
        a = a["specs"]
        b = b["specs"]
        assert len(a) == len(b), f"Specs mismatch: {len(a)} != {len(b)} in {a} vs {b}"
        for a_, b_ in zip(a, b):
            assert_compatible_specs(a_, b_)
    elif specs_is_dict(a):
        assert specs_is_dict(b), f"Specs mismatch: {a} vs {b}"
    elif specs_is_box(a):
        assert specs_is_box(b), f"Specs mismatch: {a} vs {b}"
    elif specs_is_tuple(a) or specs_is_box(a):
        a = a["specs"]
        b = b["specs"]
        assert len(a) == len(b), f"Specs mismatch: {len(a)} != {len(b)} in {a} vs {b}"
        for k in a:
            assert_compatible_specs(a[k], b[k])
    else:
        assert specs_is_final(a) and specs_is_final(b), f"Specs mismatch: {a} vs {b}. {log}"


def compare_specs(a, b):
    if isinstance(a, list):
        a = tuple(a)
    if isinstance(b, list):
        b = tuple(b)
    if isinstance(a, tuple):
        return all(compare_specs(a_, b_) for a_, b_ in zip(a, b))
    if isinstance(a, dict):
        if (specs_is_final(a) or specs_is_box(a)) and (specs_is_final(b) or specs_is_box(b)):
            return True
        if set(a.keys()) != set(b.keys()):
            return False
        for k in set(a.keys()) | set(b.keys()):
            if k not in b:
                return False
            if not compare_specs(a[k], b[k]):
                return False
        return True
    return a == b


def _apply(*args, _make_callable=False, **kwargs):
    structure = _apply_(*args, **kwargs)
    if not _make_callable:
        return structure
    specs = kwargs.get("specs")
    return _structure_to_callable(structure, specs=specs)


def _apply_(func, args, kwargs, _apply_on_boxes, no_recurse=[], specs=None):
    # print(f"DEBUG: Applying {func.__name__} with args={args}, kwargs={kwargs}, no_recurse={no_recurse}, specs={specs}")
    if specs is None:
        specs = _guess_specs(*args, kwargs=kwargs, no_recurse=no_recurse)
    check_specs(specs)
    # print("DEBUG", repr_specs(specs))

    if specs_is_final(specs):
        return func(*args, **kwargs)

    if _apply_on_boxes and specs_is_box(specs):
        return func(*args, **kwargs)

    def recurse(v, name, key):
        # pick key in dict only if argument is recursable
        return v[key] if name not in no_recurse else v

    def next_apply(key):
        return _apply_(
            func,
            args=[recurse(v, i, key) for i, v in enumerate(args)],
            kwargs={k_: recurse(v, k_, key) for k_, v in kwargs.items()},
            no_recurse=no_recurse,
            _apply_on_boxes=_apply_on_boxes,
            specs=specs["specs"][key],
        )

    if specs_is_dict(specs) or (specs_is_box(specs) and not _apply_on_boxes):
        keys = specs["specs"].keys()
        for k, v in _recursable_arguments(args, kwargs, no_recurse):
            if not isinstance(v, dict):
                raise ValueError(f"Expected dict content for argument {k}, got {type(v)}. {specs}")
            if set(v.keys()) != set(keys):
                raise ValueError(f"Keys mismatch for {k}: {set(v.keys())} != {set(keys)}")
        return {key: next_apply(key) for key in keys}

    if specs_is_tuple(specs):
        length = len(specs["specs"])
        for k, v in _recursable_arguments(args, kwargs, no_recurse):
            if not isinstance(v, (list, tuple)):
                raise ValueError(f"Expected list or tuple content for argument {k}, got {type(v)}. {specs}")
            if len(v) != length:
                raise ValueError(f"Length mismatch for {k}: {len(v)} != {length} in {specs}")
        return tuple([next_apply(key) for key in range(length)])
    raise ValueError(f"Unknown specs type: {specs}")


# def function_on_box(*args=None,**options):
#    return apply_on_box(specs_or_callable, **options)
#
# def function_on_leaf(specs_or_callable=None,**options):
#    raise NotImplementedError("TODO")


def _structure_to_callable(structure, specs=None):
    """From a structure with all leaves callable, create a function to be apply to strctures with same schema"""
    if specs is None:
        specs = guess_specs(structure)

    @apply_on_leaf
    def assert_is_callable(x):
        assert callable(x), f"Expected callable, got {type(x)}, in {repr(structure)}"

    assert_is_callable(structure)

    def func(*args, **kwargs):

        @apply_on_leaf
        def func_(x, *args, **kwargs):
            return x(*args, **kwargs)

        return func_(structure, *args, **kwargs)

    func._anemoi = dict(structure=structure, specs=specs)
    return func


def function_on_box(specs_or_callable=None, **options):
    return _apply_on_x(specs_or_callable, **options, _apply_on_boxes=True, _make_callable=True)


def function_on_leaf(specs_or_callable=None, **options):
    return _apply_on_x(specs_or_callable, **options, _apply_on_boxes=False, _make_callable=True)


def apply_on_leaf(specs_or_callable=None, **options):
    return _apply_on_x(specs_or_callable, **options, _apply_on_boxes=False, _make_callable=False)


def apply_on_box(specs_or_callable=None, **options):
    return _apply_on_x(specs_or_callable, **options, _apply_on_boxes=True, _make_callable=False)


def _apply_on_x(specs_or_callable=None, **options):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return _apply(func, args, kwargs, **options)

        return wrapper

    # decorator called with no parameters, apply the wrapper around the first arg
    no_arguments = set(options.keys()) == {"_apply_on_boxes", "_make_callable"}
    if no_arguments and callable(specs_or_callable):
        return inner(specs_or_callable)

    # decorator called with parameters, if present, first arg is the specs
    specs = specs_or_callable
    if options.get("specs") is not None and specs is not None:
        raise ValueError("Multiple values for argument 'specs'.")
    options["specs"] = specs

    return inner


class Structure:
    def __init__(self, content, specs=None):
        self.content = content
        self.specs = specs

    def __call__(self, *args, **kwargs):
        @apply_on_leaf()
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
            fields:
              tensor:
                - variables: ["era5.2t", "era5.10u", "era5.10v"]
                - offset: ["-6h"]
            other_fields:
              tensor:
                - offset: ["-6h", "+6h"]
                - variables: ["era5.2t", "era5.10u"]
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

    specs = sp.dataspecs
    print(specs)
    print(repr_specs(specs))
    # print(repr(specs, _specs=specs, name="Specs"))

    data = sp[1]
    print(data)
    print(repr_box(data, name="Data"))
    guessed = guess_specs(data)
    print(f"Guessed specs: {repr_specs(guessed)}")
    print("Specs=", repr_specs(specs))
    assert_compatible_specs(specs, guessed)

    print("----- select -------")

    @apply_on_box(specs, no_recurse=["key"])
    def select(content, key):
        return content[key]

    latitudes = select(content=data, key="latitudes")
    print(repr(latitudes, name="Lat"))
    longitudes = select(data, key="longitudes")
    print(repr(longitudes, name="Long"))
    print(guess_specs(longitudes))

    print("----- merge -------")

    # @apply_on_box(specs, no_recurse=["**kwargs"]) # not implemented
    @apply_on_box
    def merge(**kwargs):
        return dict(**kwargs)

    print(
        repr_box(
            merge(data=data, lat=latitudes),
            name="merge(data=data, lat=latitudes)",
        ),
    )

    @apply_on_leaf
    def merge_element(**kwargs):
        return dict(**kwargs)

    print(
        repr_box(
            merge_element(latitudes=latitudes, longitudes=longitudes),
            name="merge_element(latitudes=latitudes, longitudes=longitudes)",
        ),
    )

    print("------")

    @function_on_box
    def build_normaliser(statistics):
        mean = np.mean(statistics["longitudes"])
        # mean = statistics["mean"]

        def func(box):
            box['data'] = box['data'] - mean
            return box
            # return {apply_on_key: box[apply_on_key] - mean}

        # norm = Normaliser(mean)
        # def f(x):
        #    return norm.transform(x["data"])

        return func

    normaliser = build_normaliser(data)
    print(repr_box(data))
    print(repr(normaliser(data), name="normaliser(data)"))
    # print(repr(normaliser(data, apply_on_key="data"), name="normaliser(data)"))

    exit()

    @apply_on_leaf
    def apply_func(func, data):
        return func(data)

    print(
        repr(
            apply_func(normalisers, data=data),
            name="apply_func(normalisers, data=data)",
        ),
    )
    print(sp.static_info)
    print(guess_specs(sp.static_info))
    print(repr_box(sp.static_info, name="sp.static_info"))


if __name__ == "__main__":
    test(False, False, True)
