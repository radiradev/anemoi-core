# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import datetime
import os

import numpy as np
from rich.console import Console
from rich.tree import Tree

from anemoi.utils.dates import frequency_to_string

# more imports at the end of this file

ICON_BOX = "📦"
ICON_LEAF = "🌱"
ICON_LEAF_BOX_NOT_FOUND = "🍀"


def _choose_icon(k, v):
    if k.startswith("_"):
        return "  "
    return dict(
        latitudes="🌍",
        longitudes="🌍",
        timedeltas="🕐",
        data="🔢",
        statistics="  ",
        name_to_index="  ",
        normaliser="  ",
        inputer="  ",
        extra="  ",
    ).get(k, ICON_LEAF)


def format_shorten(k, v):
    if len(str(v)) < 50:
        return f"{k}: {v}"
    return f"{k}: {str(v)[:50]}..."


def format_timedeltas(k, v):
    def _to_str(x):
        x = int(x)
        x = datetime.timedelta(seconds=x)
        return frequency_to_string(x)

    try:
        if isinstance(v, np.ndarray):
            minimum = _to_str(np.min(v))
            maximum = _to_str(np.max(v))
            return f"{k}: np.array of shape {v.shape} [{minimum},{maximum}]"

        import torch

        if isinstance(v, torch.Tensor):
            minimum = _to_str(torch.min(v).item())
            maximum = _to_str(torch.max(v).item())
            return f"{k}: tensor of shape {v.shape} [{minimum},{maximum}]"

        return f"{k}: no-min, no-max"

    except (ValueError, ImportError):
        return f"{k}: [no-min, no-max]"


def format_array(k, v):
    try:
        if isinstance(v, np.ndarray):
            minimum = np.min(v)
            maximum = np.max(v)
            return f"{k}: np.array of shape {v.shape}, min/max={minimum:.2f}/{maximum:.3f}"

        import torch

        if isinstance(v, torch.Tensor):
            minimum = torch.min(v).item()
            maximum = torch.max(v).item()
            shape = ", ".join(str(dim) for dim in v.size())
            return f"{k} : tensor of shape ({shape}) on {v.device}, min/max={minimum:.3f}/{maximum:.3f}"

        return f"{k}: no-min, no-max"

    except (ValueError, ImportError):
        return f"{k}: [no-min, no-max]"


def format_key_value(k, v):
    if k == "timedeltas":
        return format_timedeltas(k, v)
    if isinstance(v, (int, float, bool)):
        return f"{k}: {v}"
    if isinstance(v, str):
        return f"{k}: '{v}'"
    if isinstance(v, (list, tuple)):
        return format_shorten(k, str(v))
    if isinstance(v, dict):
        keys = ",".join(f"{k_}" for k_ in v.keys())
        return format_shorten(k, "dict with keys " + keys)
    if isinstance(v, np.ndarray):
        return format_array(k, v)
    try:
        import torch

        if isinstance(v, torch.Tensor):
            return format_array(k, v)
    except ImportError:
        pass
    return f"{k}: {v.__class__.__name__}"


def format_tree(key, value, boxed=True):
    """Recursively build a Tree from any nested structure."""
    from anemoi.training.data.refactor.structure import is_box
    from anemoi.training.data.refactor.structure import is_final
    from anemoi.training.data.refactor.structure import is_schema

    if is_schema(value):
        return format_schema(key, value)

    if is_final(value):
        return Tree(f"{ICON_LEAF_BOX_NOT_FOUND} {key} : {value}" if key is not None else str(value))

    if is_box(value):
        if boxed:
            key = f"📦 {key}"
        t = Tree(f"{key} :")

        def priority(k):
            if k.startswith("_"):
                return 10
            return dict(latitudes=1, longitudes=2, timedeltas=3).get(k, 0)

        order = sorted(value.keys(), key=priority)
        assert len(order) == len(value), (order, value.keys())

        for k in order:
            v = value[k]
            if not os.environ.get("DEBUG") and k.startswith("_"):
                continue
            txt = format_key_value(k, v)
            if txt is not None:
                t.add(f"{_choose_icon(k, v)} {txt}")
        return t

    if isinstance(value, dict):  # must be after is_box because box is a dict
        t = Tree(str(key) if key is not None else "")
        for k, v in value.items():
            # k = "🔑 " + k
            t.add(format_tree(k, v, boxed=boxed))
        return t

    if isinstance(value, (list, tuple)):
        t = Tree(str(key) if key is not None else "")
        for i, v in enumerate(value):
            t.add(format_tree("#" + str(i), v, boxed=boxed))
        return t

    raise ValueError(f"Unknown type for value: {type(value)}. Key: {key}, Value: {value}")


def format_schema(key, value):
    if not isinstance(value, dict):
        raise ValueError(f"Expected dict for schema for {key=}, got {type(value)}: {value}")
    if "type" not in value:
        raise ValueError(f"Schema must contain 'type' key, for {key=}: {value}, got {value.keys()}")
    if "content" not in value:
        raise ValueError(f"Schema must contain 'content' key, for {key=}: {value}")

    type_ = value["type"]
    if type_ == "box":
        t = Tree(f"{ICON_BOX} {key} :")
        for k, v in value["content"].items():
            if k == "_anemoi_schema":
                continue
            t.add(format_schema(k, v))
        return t

    if type_ == "dict":
        t = Tree(str(key))
        for k, v in value["content"].items():
            if k == "_anemoi_schema":
                continue
            t.add(format_schema(k, v))
        return t

    if type_ == "tuple":
        t = Tree(str(key))
        for i, v in enumerate(value["content"]):
            if k == "_anemoi_schema":
                continue
            t.add(format_schema("#" + str(i), v))
        return t

    return Tree(f"{ICON_LEAF} {key} : {type_}")


def to_str(nested, name, _boxed=True):
    if callable(nested) and hasattr(nested, "_anemoi_function_str"):
        return name + nested._anemoi_function_str
    tree = format_tree(name, nested, boxed=_boxed)
    return _tree_to_string(tree).strip()


def _tree_to_string(tree):
    console = Console(record=True, width=120)
    with console.capture() as capture:
        console.print(tree)
    return capture.get()
