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
import torch
from rich.console import Console
from rich.tree import Tree

from anemoi.utils.dates import frequency_to_string

# more imports at the end of this file

ICON_BOX = "📦"
ICON_LEAF = "🌱"
ICON_LEAF_BOX_NOT_FOUND = "🍀"


def choose_icon(k, v):
    if str(k).startswith("_"):
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
        rollout="♻️ ",
    ).get(k, ICON_LEAF)


def format_shorten(k, v):
    if len(str(v)) < 50:
        return f"{v}"
    return f"{str(v)[:50]}..."


def format_timedeltas(k, v):
    def _to_str(x):
        x = int(x)
        x = datetime.timedelta(seconds=x)
        return frequency_to_string(x)

    try:
        if isinstance(v, np.ndarray):
            minimum = _to_str(np.min(v))
            maximum = _to_str(np.max(v))
            return f"np.array{v.shape} [{minimum},{maximum}]"

        if isinstance(v, torch.Tensor):
            minimum = _to_str(torch.min(v).item())
            maximum = _to_str(torch.max(v).item())
            return f"tensor({v.shape}) [{minimum},{maximum}]"

        return "no-min, no-max"

    except (ValueError, ImportError):
        return "[no-min, no-max]"


def format_array(k, v):
    try:
        if isinstance(v, np.ndarray):
            minimum = np.min(v)
            maximum = np.max(v)
            mean = np.nanmean(v)
            stdev = np.nanstd(v)
            return f"np.array{v.shape} {mean:.5f}±{stdev:.1f}"  # [{minimum:.1f},{maximum:.1f}]"

        import torch

        if isinstance(v, torch.Tensor):
            shape = ", ".join(str(dim) for dim in v.size())
            v = v[~torch.isnan(v)].flatten()
            if v.numel() == 0:
                minimum = float("nan")
                maximum = float("nan")
                mean = float("nan")
                stdev = float("nan")
            else:
                minimum = torch.min(v).item()
                maximum = torch.max(v).item()
                mean = torch.mean(v.float()).item()
                stdev = torch.std(v.float()).item()
            return f"tensor({shape}) on {v.device}, {mean:.5f}±{stdev:.1f}[{minimum:.1f}/{maximum:.1f}]"

        return "no-min, no-max"

    # except (ValueError, ImportError, RuntimeError):
    #    return f"{k}: [no-min, no-max]"
    except Exception as e:
        return f"[error: {e!s}]"


def format_key_value(k, v):
    txt = f"{v.__class__.__name__}"

    if k == "timedeltas":
        txt = format_timedeltas(k, v)
    elif k == "rollout_usage":
        txt = ""
        for step, lst in v.items():
            usages = ",".join(f"{kind}({when})" for kind, when in lst)
            txt += f"step{step}:{usages} "

    elif isinstance(v, (int, float, bool)):
        txt = f"{v}"
    elif isinstance(v, str):
        txt = f"'{v}'"
    elif isinstance(v, (list, tuple)):
        txt = format_shorten(k, str(v))
    elif isinstance(v, dict):
        keys = ",".join(f"{k_}" for k_ in v.keys())
        txt = format_shorten(k, "dict with keys " + keys)
    elif isinstance(v, np.ndarray):
        txt = format_array(k, v)
    try:
        import torch

        if isinstance(v, torch.Tensor):
            txt = format_array(k, v)
    except ImportError:
        pass
    if isinstance(v, np.datetime64):
        txt = f'np.datetime64("{v!s}")'
    if isinstance(v, datetime.datetime):
        txt = f"datetime({v})"
    from anemoi.training.data.refactor.sample_provider import Rollout

    if isinstance(v, Rollout):
        txt = str(v)
    return txt


def format_tree(key, value, boxed=True, **kwargs):
    """Recursively build a Tree from any nested structure."""
    assert False, "dead code?"
    from anemoi.training.data.refactor.structure import Box

    #    if is_schema(value):
    #        return format_schema(key, value)

    if isinstance(value, Box):
        if not all(isinstance(k, (int, str)) for k in value.keys()):
            print(Box)
            raise ValueError(f"Invalid keys in Box: {value.keys()}")
        if boxed:
            key = f"📦 {key}"
        t = Tree(f"{key} :", **kwargs)

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
                txt = f"{txt}"
                t.add(txt)
        return t

    if isinstance(value, (dict, torch.nn.ModuleDict)):  # must be after is_box because box is a dict
        t = Tree(str(key) if key is not None else "", **kwargs)
        for k, v in value.items():
            # k = "🔑 " + k
            t.add(format_tree(k, v, boxed=boxed, **kwargs))
        return t

    if isinstance(value, (list, tuple)):
        t = Tree(str(key) if key is not None else "", **kwargs)
        for i, v in enumerate(value):
            t.add(format_tree("#" + str(i), v, boxed=boxed, **kwargs))
        return t

    raise ValueError(f"Unknown type for value: {type(value)}. Key: {key}")


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
            __type = v.get("type")
            t.add(Tree(f"{k} : {__type}"))
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


def to_str(nested, name, _boxed=True, **kwargs):
    nested = nested.as_nested()
    tree = format_tree(name, nested, boxed=_boxed, **kwargs)
    return _tree_to_string(tree).strip()


def _tree_to_string(tree):
    console = Console(record=True, width=120)
    with console.capture() as capture:
        console.print(tree)
    return capture.get()
