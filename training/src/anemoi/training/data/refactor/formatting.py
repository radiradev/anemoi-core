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

from anemoi.utils.dates import frequency_to_string

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


def anemoi_dict_to_str(obj):
    assert isinstance(obj, dict), type(obj)
    from anemoi.training.data.refactor.structure import Dict

    assert isinstance(obj, Dict), type(obj)

    if not obj:
        return f"{obj.__class__.__name__} (empty)"
    # this function is quite long and has a lot of knowledge about the other types
    # it is not too bad because everything related to display is here
    # but this is ugly
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
                    t.add(value.tree(prefix=key))
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

    name = obj.__class__.__name__
    for leaf in obj.values():
        if isinstance(leaf, dict) and "_reference_date" in leaf:
            name += f" (Reference {leaf['_reference_date']})"
            break
    tree = Tree(name)

    verbose = int(os.environ.get("ANEMOI_CONFIG_VERBOSE_STRUCTURE", 0))
    leaf_to_tree = {0: one_line_leaf, 1: expanded_leaf, 2: debug_leaf}[verbose]
    for path, leaf in obj.items():
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
