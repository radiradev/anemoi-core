# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations


from boltons.iterutils import research as _research  # noqa: F401

SEPARATOR = "__"

MAPPING_OF_ALLOWED_CHARACTERS_IN_DICT_KEYS = {
    # "Xbackslash_": "\\",
    # "Xbar_": "|",
    "Xcolon_": ":",
    # "Xequal_": "=",
    "Xgreater_": "<",
    "Xless_": ">",
    "Xminus_": "-",
    "Xplus_": "+",
    # "Xquestion_": "?",
    # "Xslash_": "/",
    "Xstar_": "*",
    # "Xtilde_": "~",
}
ALLOWED_CHARACTERS_IN_DICT_KEYS = set(MAPPING_OF_ALLOWED_CHARACTERS_IN_DICT_KEYS.values())


def _check_no_conflict(dic):
    for key in dic:
        conflicts = {k: v for k, v in dic.items() if k.startswith(key)}
        if len(conflicts) > 1:
            raise ValueError(f"Conflict in allowed characters mapping: {conflicts}")


_check_no_conflict(MAPPING_OF_ALLOWED_CHARACTERS_IN_DICT_KEYS)


def check_dictionary_key(k):
    if not isinstance(k, str):
        raise TypeError(f"Expected string for dictionary key, got {type(k)}: {k}")
    if SEPARATOR in k:
        raise ValueError(f"Keys in dictionary must not contain '{SEPARATOR}', got: {k}")
    if k.startswith("_"):
        raise ValueError(f"Keys in dictionary must not start with '_', got: {k}")
    for c in k:
        if not c.isalnum() and c not in ALLOWED_CHARACTERS_IN_DICT_KEYS and c not in ["_", "X"]:
            raise ValueError(
                f"Keys in dictionary must only contain {ALLOWED_CHARACTERS_IN_DICT_KEYS} characters and _, got: '{c}' in '{k}'",
            )

    if k.lower() != k:
        # we could allow capital letters and encode the case in the mapping
        # but the encoded keys would be hard to read
        # and this would need careful handling
        raise ValueError(f"Keys in dictionary must be lowercase, got: {k}.")

    # if k[0].isdigit():
    # this is to keep open the possibility to handle lists/tuples in the future
    #    raise ValueError(f"Keys in dictionary must not start with a digit for now, got: {k}")
    RESERVED_WORDS_FOR_TORCH_MODULE = [  # from torch.nn.Module
        "dump_patches",
        "call_super_init",
        "forward",
        "register_buffer",
        "register_parameter",
        "add_module",
        "register_module",
        "get_submodule",
        "set_submodule",
        "get_parameter",
        "get_buffer",
        "get_extra_state",
        "set_extra_state",
        "apply",
        "cuda",
        "ipu",
        "xpu",
        "mtia",
        "cpu",
        "type",
        "float",
        "double",
        "half",
        "bfloat16",
        "to_empty",
        "to",
        "register_full_backward_pre_hook",
        "register_backward_hook",
        "register_full_backward_hook",
        "register_forward_pre_hook",
        "register_forward_hook",
        "register_state_dict_post_hook",
        "register_state_dict_pre_hook",
        "T_destination",
        "state_dict",
        "register_load_state_dict_pre_hook",
        "register_load_state_dict_post_hook",
        "load_state_dict",
        "parameters",
        "named_parameters",
        "buffers",
        "named_buffers",
        "children",
        "named_children",
        "modules",
        "named_modules",
        "train",
        "eval",
        "requires_grad_",
        "zero_grad",
        "share_memory",
        "extra_repr",
        "compile",
    ]
    if k in RESERVED_WORDS_FOR_TORCH_MODULE:
        raise ValueError(
            f"Keys in dictionary must not be a reserved word for torch.nn.Module, got: {k}. Recommending to use '{k}_' instead",
        )
    try:
        import torch

        for cls in [
            torch.nn.Module,
            torch.nn.ModuleDict,
            torch.nn.ParameterDict,
            torch.nn.ModuleList,
            torch.nn.ParameterList,
        ]:
            if k in cls.__dict__:
                raise ValueError(f"Keys in dictionary must not be a {cls.__name__} attribute, got: {k}")
    except ImportError:
        pass

    return k


def _path_as_str(path):
    if isinstance(path, (list, tuple)):
        return SEPARATOR.join(_path_as_str(x) for x in path)
    if not isinstance(path, str):
        raise KeyError(f"Path must be str, list or tuple, got {type(path)}")
    if path.startswith("."):
        raise KeyError(f"Path starting with {SEPARATOR} is not allowed. Got {path}")

    for k, v in MAPPING_OF_ALLOWED_CHARACTERS_IN_DICT_KEYS.items():
        if v in path:
            path = path.replace(v, k)

    path = path.replace(".", SEPARATOR)
    check_path(path)
    return path


def check_path(path):
    for c in path:
        _check_path_character(c, path)


def _check_path_character(c, path):
    # should be next to check_dictionary_key
    if c == ".":
        raise KeyError(f"Path cannot contain '.', got {path}")
    for allowed in ALLOWED_CHARACTERS_IN_DICT_KEYS:
        if c == allowed:  # should have been converted
            raise KeyError(f"Path cannot contain '{c}', got {path}", ALLOWED_CHARACTERS_IN_DICT_KEYS)
    if c in [SEPARATOR, "X", "_"]:
        return
    if c.isupper():
        raise KeyError(f"Path cannot contain uppercase letters, got {path}")


def _join_paths(path1, path2):
    return SEPARATOR.join([path1, path2])


def _path_as_tuple(path):
    if isinstance(path, str):
        return tuple(int(x) if x.isdigit() else x for x in path.split(SEPARATOR))
    if isinstance(path, int):
        return (path,)
    if isinstance(path, tuple):
        return path
    if isinstance(path, list):
        return tuple(path)
    raise ValueError(f"Path must be str, int, list or tuple, got {type(path)}")
