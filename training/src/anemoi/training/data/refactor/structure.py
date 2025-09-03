# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from functools import wraps

import boltons
import numpy as np
import yaml
from boltons.iterutils import default_enter as _default_enter
from boltons.iterutils import default_exit as _default_exit
from boltons.iterutils import get_path as _get_path
from boltons.iterutils import remap as _remap
from rich import print

from anemoi.training.data.refactor.formatting import to_str
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

# from boltons.iterutils import research as _research,


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


class Box(dict):
    # Flag a dict as a box
    pass


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
            res.update(box)
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
        raise Exception("apply_to_box decorator takes at most one non-keyword argument : the function to wrap.")

    @wraps(constructor)
    def wrapper(fconfig_n, *fargs, **fkwargs):
        return _for_each_expanded_box(fconfig_n, constructor, *fargs, **fkwargs)

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
    return root


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
                input:
                    dictionary:
                        fields:
                            container:
                              data_group: "era5"
                              variables: ["2t", "10u", "10v"]
                              dimensions: ["values", "variables", "ensembles"]
                        other_fields:
                          for_each:
                            - offset: ["-6h", "0h"]
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


def test_two(training_context):
    from rich.console import Console
    from rich.table import Table

    console = Console()

    from anemoi.training.data.refactor.sample_provider import sample_provider_factory

    def print_columns(*args):
        console = Console()
        table = Table(show_header=False, box=None)
        for a in args:
            table.add_column()
        table.add_row(*args)
        console.print(table)

    print("✅✅✅✅✅✅✅✅✅✅✅✅✅✅")
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
    print(to_str(sp.static_info, "Static Info"))
    data = sp[1]
    print(to_str(data, "Full Data"))
    data = merge_boxes(data, sp.static_info)

    def i_to_delta(i, frequency):
        frequency = frequency_to_timedelta(frequency)
        delta = frequency * i
        sign = "+" if i > 0 else ""
        if delta:
            return sign + frequency_to_string(delta)
        return "0h"

    data = select_content(data, "data", "_offset")
    # data = select_content(data, "_offset")
    # def change_source(dic, key, source):
    #    path = dic[key].split('.')
    #    path[0] = source
    #    dic[key] = '.'.join(path)
    # if i == 0:
    #    change_source(rollout_config[0]["input"], "prognostics.0", source='from_dataset')

    console.print("[red] ✅ Rollout for prognostic data[/red]")
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
        input = input_target["input"]
        target = input_target["target"]
        print_columns(to_str(input, f"input({i})"), to_str(target, f"target({i})"))

        output = input_target["target"]
        previous_input = input_target["input"]


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

    training_context = dict(
        sources=sources_config,
        start=None,
        end=None,
        frequency="6h",
    )

    ONE = "1" in sys.argv
    TWO = "2" in sys.argv
    THREE = "3" in sys.argv
    if ONE:
        test_one(training_context)

    if TWO:
        test_two(training_context)


if __name__ == "__main__":

    test()
