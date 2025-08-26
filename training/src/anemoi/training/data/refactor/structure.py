# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import json
from functools import wraps

import numpy as np
import yaml
from rich import print

from anemoi.training.data.refactor.formatting import to_str


class Box(dict):
    # Flag a dict as a box
    pass


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


from boltons.iterutils import default_enter as _default_enter
from boltons.iterutils import default_exit as _default_exit
from boltons.iterutils import get_path as _get_path
from boltons.iterutils import remap as _remap

# research as _research,

ICON_BOX = "📦"
ICON_LEAF = "🌱"
ICON_LEAF_BOX_NOT_FOUND = "🍀"


def _check_key_in_dict(dic, key):
    if not isinstance(dic, dict):
        raise ValueError(f"Expected dict, got {type(dic)}")
    if key not in dic:
        raise ValueError(f"Key {key} not found in dict. Available keys are: {list(dic.keys())}")


def is_final(structure):
    return not isinstance(structure, (dict, list, tuple))


def is_box(structure):
    if isinstance(structure, Box):
        return True
    if not isinstance(structure, dict):
        return False
    # Not so reliable, see inside the dict if there is a final element
    return any(is_final(v) for v in structure.values())


def is_schema(x):
    if not isinstance(x, dict):
        return False
    return "_anemoi_schema" in x


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
        for other_structure in structs:
            other_box = _get_path(other_structure, path + (key,), default={})
            if not overwrite and set(other_box.keys()) & set(res.keys()):
                raise ValueError(f"Conflicting keys found in structures: {set(other_box.keys()) & set(res.keys())}")
            res.update(other_box)
        return res

    return _remap(structs[0], exit=exit)


def _stop_if_box_enter(path, key, value):
    if is_box(value):
        return value, []
    return _default_enter(path, key, value)


def _for_each_expanded_box(fconfig_tree, constructor, *args, **kwargs):
    # apply the constructor to each box in the fconfig_tree tree

    def transform(path, key, fconfig_box):
        if not is_box(fconfig_box):
            # apply only on boxes
            return key, fconfig_box
        if any(callable(v) for k, v in fconfig_box.items()):
            return key, fconfig_box

        # print(f"DEBUG: {path} {key}, applying function {constructor.__name__} to {value} {args=} {kwargs=}")
        value = constructor(**fconfig_box, **kwargs)
        # print(f"  -> {value=}")

        # if we wanted to output boxes : return key, dict(function=value)
        return key, value

    # return a structure where each box is replaced by a function
    return _remap(fconfig_tree, visit=transform, enter=_stop_if_box_enter)


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
def box_to_function(constructor, **options):
    if options or not callable(constructor):
        raise Exception("box_to_function decorator takes at most one non-keyword argument, the function to wrap.")

    @wraps(constructor)
    def wrapper(fconfig_n, *fargs, **fkwargs):
        res_n = _for_each_expanded_box(fconfig_n, constructor, *fargs, **fkwargs)
        return nested_to_callable(res_n)

    return wrapper


# decorator
def apply_to_box(constructor, **options):
    if options or not callable(constructor):
        raise Exception("apply_to_box decorator takes at most one non-keyword argument, the function to wrap.")

    @wraps(constructor)
    def wrapper(fconfig_n, *fargs, **fkwargs):
        res_n = _for_each_expanded_box(fconfig_n, constructor, *fargs, **fkwargs)
        return res_n

    return wrapper


# decorator
def make_output_callable(func):
    if not callable(func):
        raise Exception("make_output_callable decorator takes a callable argument, the function to wrap.")

    @wraps(func)
    def wrapper(*args, **kwargs):
        res_tree = func(*args, **kwargs)
        return nested_to_callable(res_tree)

    return wrapper


def nested_to_callable(f_tree):
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
    # print(repr_schema(schema, "schema"))

    print(to_str(sp.static_info, name="✅ sp.static_info"))
    print(sp.static_info)

    data = sp[1]
    print(to_str(data, name="✅ Data"))
    print(data)

    guessed = make_schema(data)
    to_str(schema, "Actual Schema")
    to_str(guessed, "Actual Schema")
    # print(to_str(guessed, "Guessed Schema"))
    # print(to_str(schema, "Actual Schema"))
    # print(to_str(guessed, "Guessed Schema"))
    # print(schema)
    # assert_compatible_schema(schema, guessed)

    data = merge_boxes(data, sp.static_info)
    print(to_str(data, name="✅ Data merged with sp.static_info"))

    extra = remove_content(data, "extra")
    print(to_str(data, name="Data after removing extra"))
    print(to_str(extra, name="Extra content removed from data"))

    lat_lon = select_content(data, "latitudes", "longitudes")
    print(to_str(lat_lon, name="Selected latitudes and longitudes"))

    @box_to_function
    def build_normaliser(statistics, **kwargs):
        mean = np.mean(statistics["mean"])

        def func(data, **kwargs):
            new = data - mean
            return dict(normalized_data=new, **kwargs)

        # norm = Normaliser(mean,...)
        # def f(x):
        #    return norm.transform(x["data"])

        return func

    print(to_str(sp.static_info, name="sp.static_info"))
    normaliser = build_normaliser(sp.static_info)
    # normaliser = nested_to_callable(normaliser)
    # n_data = apply(normaliser,data)
    n_data = normaliser(data)
    print(to_str(normaliser, name="Normaliser function"))
    print(to_str(data, name="data before normalisation"))
    print(to_str(n_data, name="normaliser(data)"))
    print(n_data)


if __name__ == "__main__":
    test(False, False, True)
