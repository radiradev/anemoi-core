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
import warnings
from abc import abstractmethod
from functools import cached_property

import einops
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from rich.console import Console
from rich.tree import Tree

from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

from .data_handler import DataHandler


def resolve_reference(config):
    from omegaconf import OmegaConf

    config = OmegaConf.create(config)
    config = OmegaConf.to_container(config, resolve=True)
    return config


def seconds(td):
    return round(td.total_seconds())


class Context:
    start = None
    end = None
    frequency = None
    sources = None
    _offset = None
    _all_nodes = None
    _visible_nodes = None
    _root = None

    def __init__(
        self,
        start=None,
        end=None,
        frequency=None,
        sources=None,
        offset=None,
        _parent=None,
    ):
        self._parent = _parent
        self._copy_from_parent(_parent)

        if offset is None:
            offset = "0h"
        if isinstance(offset, str):
            offset = frequency_to_timedelta(offset)
        self._offset = self._offset + offset if self._offset else offset

        self.start = self.start if start is None else start
        self.end = self.end if end is None else end

        self.frequency = self.frequency if frequency is None else frequency_to_timedelta(frequency)

        self._all_nodes = self._all_nodes if self._all_nodes is not None else []
        self._visible_nodes = self._visible_nodes if self._visible_nodes is not None else []

        self.sources = self.sources if sources is None else resolve_reference(sources)
        assert self.sources is not None, f"Sources cannot be None, got {self.sources}"

        def processor_factory(config, name_to_index=None, statistics=None):
            return instantiate(
                config,
                name_to_index_training_input=name_to_index,
                statistics=statistics,
            )

        self.processor_factory = processor_factory

        assert isinstance(
            self.offset,
            datetime.timedelta,
        ), f"Expected timedelta for offset, got {type(self.offset)}: {self.offset}"

    def _copy_from_parent(self, _parent):
        if _parent is None:
            return
        if not isinstance(_parent, Context):
            raise TypeError(f"Expected Context as parent, got {type(_parent)}: {_parent}")
        self.start = _parent.start
        self.end = _parent.end
        self.frequency = _parent.frequency
        self.sources = _parent.sources
        self._offset = _parent._offset
        self._all_nodes = _parent._all_nodes
        self._visible_nodes = _parent._visible_nodes
        self._root = _parent._root

    @property
    def offset(self):
        if self._offset is None:
            return frequency_to_timedelta("0h")
        return self._offset

    def register(self, obj):
        if not isinstance(obj, SampleProvider):
            raise TypeError(f"Expected SampleProvider, got {type(obj)}: {obj}")
        assert obj not in self._all_nodes, f"Object {obj} is already registered in {self._all_nodes}"

        self._all_nodes.append(obj)

    def register_visible(self, obj):
        assert obj not in self._visible_nodes, f"Object {obj} is already registered in {self._visible_nodes}"
        self._visible_nodes.append(obj)

    def __repr__(self):
        return f"Context(start={self.start}, end={self.end}, offset={self.offset})"


class VariablesList:
    def __init__(self, variables: list[str] | dict, group):
        self.lst = [f"{group}.{v}" for v in variables]
        return

        if isinstance(variables, dict):
            self.lst = []
            for group, vars_ in variables.items():
                if isinstance(vars_, str):
                    vars_ = [vars_]
                if not isinstance(vars_, (list, tuple)):
                    raise ValueError(f"Expected list or tuple for variables, got {type(vars_)}: {vars_}")
                for v in vars_:
                    if not isinstance(v, str):
                        raise ValueError(f"Expected string for variable, got {type(v)}: {v}")
                    if "." in v:
                        raise ValueError(
                            f"Variable '{v}' should not contain a group name ('.' expected) in {variables})",
                        )
                self.lst += [f"{group}.{v}" for v in vars_]
            return

        if not isinstance(variables, (list, tuple)):
            raise ValueError(f"Expected list or tuple for variables, got {type(variables)}: {variables}")

        for v in variables:
            if not isinstance(v, str):
                raise ValueError(f"Expected string for variable, got {type(v)}: {v} in {variables}")
            if "." not in v:
                raise ValueError(f"Variable '{v}' does not contain a group name ('.' expected) in {variables})")
        self.lst = variables

    @property
    def as_list(self):
        return self.lst

    @cached_property
    def as_dict(self):
        dic = {}
        for v in self.lst:
            group, var = v.split(".", 1)
            if group not in dic:
                dic[group] = []
            dic[group].append(var)
        return dic

    def __repr__(self):
        return f"({', '.join(self.lst)})"


class SampleProvider:
    min_offset = None
    max_offset = None

    def __init__(self, _context: Context, _parent):
        _context.register(self)
        self._context = _context
        self._parent = _parent
        self._frequency = _context.frequency
        self.offset = frequency_to_timedelta(_context.offset)

    # public
    @property
    def static_info(self):
        from anemoi.training.data.refactor.structure import Batch

        return Batch(self._get_static(None))

    # public
    def __getitem__(self, item):
        return self._get_item(None, item)

    # private ?
    def _get_static(self, request):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    # private ?
    def _get_item(self, request, item):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    def set_min_max_offsets(self, minimum=None, maximum=None, dropped_samples=None):
        self.min_offset = minimum
        self.max_offset = maximum
        self.dropped_samples = dropped_samples

    @property
    def is_root(self):
        return self._parent is None

    def mutate(self):
        return self

    def __len__(self):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    def latitudes(self, item: int):
        raise NotImplementedError

    def longitudes(self, item: int):
        raise NotImplementedError

    def timedeltas(self, item: int):
        raise NotImplementedError

    @property
    def dataschema(self):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    @property
    def imputer(self):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    @property
    def processors(self):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    @property
    def configs(self):
        raise ValueError("Obsolete, please use self.normaliser or self.imputer or self.extra instead")

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def frequency(self):
        return self._frequency

    def __repr__(self):
        console = Console(record=True, width=120)
        tree = self.tree()
        with console.capture() as capture:
            console.print(tree)
        return capture.get()

    def tree(self, label: str = None, **kwargs):
        raise NotImplementedError("Subclasses must implement tree method in " + self.__class__.__name__)

    def _check_item(self, item: int):
        if not isinstance(item, (int, np.integer)):
            raise TypeError(f"Not implemented for non-integer indexing {type(item)}")

    def shuffle(self, *args, **kwargs):
        # TODO: remove double self, self
        return ShuffledSampleProvider(self._context, self, self, *args, **kwargs)


class ForwardSampleProvider(SampleProvider):
    def __init__(self, _context, _parent, sample: SampleProvider):
        super().__init__(_context, _parent)
        self._forward = sample

    def __len__(self):
        return len(self._forward)

    @property
    def dataschema(self):
        return self._forward.dataschema

    @property
    def imputer(self):
        return self._forward.imputer

    @property
    def processors(self):
        return self._forward.processors

    @property
    def shape(self):
        return self._forward.shape

    @property
    def frequency(self):
        return self._forward.frequency


class ShuffledSampleProvider(ForwardSampleProvider):
    label = "Shuffled"
    emoji = "🎲"

    def __init__(self, _context, _parent, sample: SampleProvider, seed: int = None):
        super().__init__(_context, _parent, sample)
        self.seed = seed
        length = len(self._forward)
        self.idx = np.arange(length)
        if seed is not None:
            np.random.seed(seed)
        self.idx = np.random.permutation(self.idx)

    def _get_item(self, request, item: int):
        print(f"Shuffling : requested {item}, provided {self.idx[item]}")
        return self._forward._get_item(request, self.idx[item])

    def tree(self, prefix=""):
        tree = Tree(prefix + self.emoji + self.label + f" (seed={self.seed})")
        subtree = self._forward.tree()
        tree.add(subtree)
        return tree


class _LoopSampleProvider(SampleProvider):
    def __init__(self, _context: Context, _parent, for_each: dict):
        super().__init__(_context, _parent)

        if not isinstance(for_each, (list, tuple)) or not len(for_each) == 2:
            raise ValueError(f"Expected list of length 2 in for_each: loop, template. Got {len(for_each)}: {for_each}")

        loop_on, template = for_each
        if not isinstance(loop_on, dict) or not len(loop_on) == 1:
            raise ValueError(
                f"Expected dict with one key for the first element of for_each, got {type(loop_on)}: in {for_each}",
            )

        key, values = list(loop_on.items())[0]
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"Expected list/tuple to loop on values, got {type(values)}: {values} in {for_each}")

        new_config = self.new_config(key, values, template)
        self._to_mutate = _sample_provider_factory(_context, _parent=_parent, **new_config)

    @abstractmethod
    def new_config(self, key, values, template):
        pass

    def mutate(self):
        return self._to_mutate


class TupleLoopSampleProvider(_LoopSampleProvider):
    def new_config(self, key, values, template):
        return dict(tuple=[{key: value, **template} for value in values])


class DictionaryLoopSampleProvider(_LoopSampleProvider):
    def new_config(self, key, values, template):
        return dict(dictionary={str(value): {key: value, **template} for value in values})


class DictSampleProvider(SampleProvider):
    label = "Dictionary"
    emoji = "📖"

    def __init__(self, _context: Context, _parent, dictionary: dict):
        super().__init__(_context, _parent)

        if not isinstance(dictionary, dict):
            raise TypeError(f"Expected dictionary, got {type(dictionary)}: {dictionary}")
        if len(dictionary) == 0:
            raise ValueError("Dictionary is empty, cannot create sample provider.")
        for k in dictionary:
            if not isinstance(k, str):
                raise ValueError(f"Keys in dictionary must be strings, got {type(k)}, {k}")

        def check_key(k):
            if not isinstance(k, str):
                raise TypeError(f"Expected string for dictionary key, got {type(k)}: {k}")
            if k.startswith("_"):
                raise ValueError(f"Keys in dictionary must not start with '_', got: {k}")
            ALLOWED_CHARACTERS = set("_+-<=>|~")
            if not all(c.isalnum() or c in ALLOWED_CHARACTERS for c in k):
                raise ValueError(f"Keys in dictionary must only contain alphanumeric characters and +/-/=/~, got: {k}")
            return k.lower()

        for k, v in dictionary.items():
            check_key(k)
            if not isinstance(v, dict):
                raise ValueError(f"Expected dictionary for sample provider, got {type(v)}: {v}. ")

        self._samples = {k: _sample_provider_factory(_context, **v, _parent=self) for k, v in dictionary.items()}

    def _get_item(self, request, item):
        return {k: v._get_item(request, item) for k, v in self._samples.items()}

    def _get_static(self, request):
        return {k: v._get_static(request) for k, v in self._samples.items()}

    @property
    def dataschema(self):
        return dict(type="dict", content={k: v.dataschema for k, v in self._samples.items()}, _anemoi_schema=True)

    @property
    def shape(self):
        return {k: v.shape for k, v in self._samples.items()}

    def tree(self, prefix=""):
        if not hasattr(self, "_samples"):
            return Tree(prefix + self.label + "<partially-initialised>")
        tree = Tree(prefix + self.label)
        for k, v in self._samples.items():
            subtree = v.tree(prefix=f'"{k}" : ')
            tree.add(subtree)
        return tree

    def __len__(self):
        lengths = [len(s) for s in self._samples.values()]
        assert lengths, "No samples in dictionary, cannot determine length."
        assert (
            len(set(lengths)) == 1
        ), f"Samples in dictionary have different lengths: {lengths}. Cannot determine length."
        return lengths[0]


class _FilterSampleProvider(SampleProvider):
    emoji = "filter-emoji"
    label = "_Filter"

    keyword = None

    def __init__(self, _context: Context, _parent, **kwargs):
        super().__init__(_context, _parent)
        kwargs = kwargs.copy()

        assert self.keyword in kwargs, f"Keyword '{self.keyword}' not found in {kwargs}"
        self.values = kwargs.pop(self.keyword)

        new_context = Context(**{"_parent": _context, self.keyword: self.values})
        self._forward = _sample_provider_factory(new_context, **kwargs, _parent=_parent)

    @property
    def shape(self):
        return self._forward.shape

    def _get_item(self, request, item: int):
        return self._forward._get_item(request, item)

    def _get_static(self, request):
        warnings.warn("TODO: change the metadata for offseted data?")
        return self._forward._get_static(request)

    @property
    def dataschema(self):
        return self._forward.dataschema

    def tree(self, prefix: str = ""):
        tree = self._forward.tree()
        tree.label = tree.label + f" ({self.emoji} {self.values})"
        return tree

    def __len__(self):
        return len(self._forward)


class OffsetSampleProvider(_FilterSampleProvider):
    emoji = "⏱️"
    label = "Offset"
    keyword = "offset"

    def _get_static(self, request):
        static = super()._get_static(request)
        static["_offset"] = self.values
        from anemoi.training.data.refactor.structure import Box

        assert isinstance(static, Box), f"Expected Box, got {type(static)}: {static}"
        return static


class TensorReshapeSampleProvider(ForwardSampleProvider):
    emoji = "🔄"
    label = "Reshape"

    def __init__(self, _context: Context, _parent, reshape):
        reshape = reshape.copy()
        dimensions = reshape.pop("dimensions")
        sample = _sample_provider_factory(_context, _parent=self, **reshape)

        self._static_cache = sample.static_info
        self.to_dimensions_order = dimensions
        self.from_dimensions_order = self._static_cache["_dimensions_order"]

        for d in self.to_dimensions_order:
            if d not in self.from_dimensions_order:
                raise ValueError(f"Dimension '{d}' not found in dataset for {sample}")

        super().__init__(_context, _parent, sample)

    def _get_static(self, request):
        box = self._forward._get_static(request)
        from anemoi.training.data.refactor.structure import Box

        assert isinstance(box, Box), f"Expected Box, got {type(box)}: {box}"
        box = box.copy()
        assert isinstance(box, Box), f"Expected Box, got {type(box)}: {box}"
        if "shape" in box:
            from_dims = " ".join([d if d in self.from_dimensions_order else "1" for d in self.from_dimensions_order])
            to_dims = " ".join(self.to_dimensions_order)
            box["shape"] = TODO
        box["_dimensions_order"] = self.to_dimensions_order
        return box

    def _get_item(self, request, item):
        box = self._forward._get_item(request, item)
        box = box.copy()
        from_dims = " ".join([d if d in self.to_dimensions_order else "1" for d in self.from_dimensions_order])
        to_dims = " ".join(self.to_dimensions_order)
        if "data" in box:
            try:
                box["data"] = einops.rearrange(box["data"], f"{from_dims} -> {to_dims}")
            except Exception as e:
                e.add_note(f"{e} while rearranging {(box['data'].shape)} from '{from_dims}' to '{to_dims}'")
                e.add_note(f"{self}")
                raise e
        return box

    @property
    def dataschema(self):
        schema = self._forward.dataschema
        schema["_dimensions_order"] = self.to_dimensions_order
        return schema

    def shape(self):
        raise NotImplementedError("Dead code here, remove all 'shape' methods")

    def tree(self, prefix=""):
        if not hasattr(self, "to_dimensions_order"):
            return Tree(f"{prefix}{self.emoji} {self.label}")

        tree = Tree(f"{prefix}{self.emoji} {self.label} ({self.to_dimensions_order})")
        tree.add(self._forward.tree(prefix=" "))

        return tree


class BoxSampleProvider(SampleProvider):
    emoji = "📦"
    label = "Data"

    i_offset = None
    dropped_samples = None

    _mutate = None

    def __init__(self, _context: Context, _parent, container):
        box = container
        box = box.copy()
        super().__init__(_context, _parent)
        if "dimensions" in box:
            # box.pop('dimensions')
            dimensions = box.pop("dimensions")
            self._mutate = dict(reshape=dict(dimensions=dimensions, container=box))
            return

        if "data_group" not in box:
            raise ValueError(f"Expected 'data_group' in box, got {box}")
        self.data_group = box.pop("data_group")
        self.variables = box.pop("variables", None)

    def mutate(self):
        if self._mutate:
            return _sample_provider_factory(self._context, _parent=self._parent, **self._mutate)
        return self

    def _get_static(self, request):
        from anemoi.training.data.refactor.structure import Box

        return Box(self.datahandler._get_static(request))

    def _get_item(self, request, item):
        actual_item = item + self.i_offset

        if actual_item < 0 or actual_item >= len(self.datahandler):
            print("❌", self)
            msg = f"Item {item} ({actual_item}) is out of bounds with i_offset {self.i_offset}, lenght of the dataset is {len(self.datahandler)} and dropped_samples is {self.dropped_samples}."
            raise IndexError(msg)

        return self.datahandler._get_item(request, item)

    @property
    def dataschema(self):
        return self.datahandler._dataschema

    @cached_property
    def datahandler(self):
        return DataHandler(
            self.data_group,
            variables=self.variables,
            sources=self._context.sources,
        )

    def tree(self, prefix: str = ""):
        def _(x):
            return frequency_to_string(x) if x else "0h"

        string_offset = _(self.offset)
        if not self.offset:
            string_offset = "0h"

        txt = f"{prefix}{self.emoji} {self.label} "
        txt += f"group={self.data_group} "
        txt += f"variables:{','.join(self.variables)}"
        txt += f" from {self.datahandler.dataset_name}"
        # txt += f"({'/'.join(self.variables)}, offset={string_offset})"
        tree = Tree(txt)

        if os.environ.get("ANEMOI_CONFIG_VERBOSE_STRUCTURE"):
            if self.min_offset is not None:
                tree.add(f"global_min_offset={_(self.min_offset)}")
            if self.max_offset is not None:
                tree.add(f"global_max_offset={_(self.max_offset)}")
            tree.add(f"length={self.__len__()}")
            if self.i_offset is not None:
                explain = f"({_(self.offset)} - {_(self.min_offset)})/{_(self.frequency)} = {_(self.actual_offset)}/{_(self.frequency)} = {self.i_offset}"
                tree.add(f"i -> i + {self.i_offset} because {explain}")
        return tree

    def set_min_max_offsets(self, minimum=None, maximum=None, dropped_samples=None):
        self.min_offset = minimum
        self.max_offset = maximum
        self.dropped_samples = dropped_samples
        self.actual_offset = self.offset - self.min_offset

        def _(x):
            return frequency_to_string(x) if x else "0h"

        if self.frequency != self._context.frequency:
            print(f"Warning: Frequency mismatch: {_(self.frequency)} != {_(self._context.frequency)}. ")
            print(self)
            raise NotImplementedError(
                f"Frequency mismatch: {_(self.frequency)} != {_(self._context.frequency)}. "
                "For now, the frequency must match the context frequency."
                "This will be implemented in the future if needed.",
            )

        i_offset = seconds(self.actual_offset) / seconds(self.frequency)
        if i_offset != int(i_offset):
            print("❌", self)
            msg = (
                f"Offset {_(self.offset)} or {_(self.min_offset)} is not a multiple of frequency {_(self.frequency)}, for {self}. "
                f"i_offset = {i_offset} is not an integer."
            )
            raise ValueError(msg)
        self.i_offset = int(i_offset)

    def __len__(self):
        if self.dropped_samples is None:
            return None
        return len(self.datahandler) - self.dropped_samples


global SAVE_COUNTER
SAVE_COUNTER = 0


def SAVE(kwargs):
    global SAVE_COUNTER
    SAVE_COUNTER += 1
    import yaml

    with open(f"sample_provider_{SAVE_COUNTER}.yaml", "w") as f:
        f.write(yaml.dump(kwargs))


def _remove_omega_conf(x):
    assert not isinstance(x, SampleProvider), type(x)

    try:
        from omegaconf import DictConfig
        from omegaconf import ListConfig
        from omegaconf import OmegaConf

        if isinstance(x, DictConfig) or isinstance(x, ListConfig):
            warnings.warn("Received Omegaconf input for sample_provider_factory, converting to standard python types")
            x = OmegaConf.to_container(x, resolve=True)
        if isinstance(x, dict):
            x = {k: _remove_omega_conf(v) for k, v in x.items()}
        if isinstance(x, list):
            x = [_remove_omega_conf(v) for v in x]
        if isinstance(x, tuple):
            x = tuple(_remove_omega_conf(v) for v in x)
    except ImportError:
        pass

    return x


def sample_provider_factory(**kwargs):
    kwargs = _remove_omega_conf(kwargs)
    SAVE(kwargs)

    _context = Context(
        sources=kwargs.pop("sources", None),
        start=kwargs.pop("start", None),
        end=kwargs.pop("end", None),
        frequency=kwargs.pop("frequency"),
    )
    return _sample_provider_factory(**kwargs, _parent=None, _context=_context)


def _sample_provider_factory(_context=None, **kwargs):
    # print('💬', kwargs)
    initial_kwargs = kwargs
    kwargs = kwargs.copy()

    assert _context is not None
    assert "_parent" in kwargs, kwargs

    if "offset" in kwargs:
        obj = OffsetSampleProvider(_context, **kwargs)
    elif "reshape" in kwargs:
        obj = TensorReshapeSampleProvider(_context, **kwargs)
    elif "dictionary" in kwargs:
        obj = DictSampleProvider(_context, **kwargs)
    elif "tuple" in kwargs:
        kwargs["tuple_"] = kwargs.pop("tuple")
        obj = TupleSampleProvider(_context, **kwargs)
    elif "for_each" in kwargs:
        obj = DictionaryLoopSampleProvider(_context, **kwargs)
    elif "for_each_as_tuple" in kwargs:
        obj = TupleLoopSampleProvider(_context, **kwargs)
    elif "container" in kwargs:
        obj = BoxSampleProvider(_context, **kwargs)
    elif "repeat" in kwargs:
        repeat = kwargs.pop("repeat")
        obj = _sample_provider_factory(_context, **kwargs)
    elif "structure" in kwargs:
        if isinstance(kwargs["structure"], SampleProvider):
            return kwargs["structure"]  # not mutate here?: todo: think about it
        if isinstance(kwargs["structure"], DictConfig):
            kwargs["structure"] = dict(kwargs["structure"])
        if isinstance(kwargs["structure"], dict):
            obj = _sample_provider_factory(_context, **kwargs["structure"])
        else:
            raise ValueError(
                f"Expected dictionary for 'structure', got {type(kwargs['structure'])}: {kwargs['structure']}",
            )
    else:
        assert False, f"Unknown sample type for kwargs {kwargs.keys()}"
    obj_ = None
    while obj != obj_:
        obj_ = obj
        obj = obj.mutate()
    del obj_

    if obj.is_root:
        # set the min/max offsets for the root sample provider
        nodes = obj._context._all_nodes
        all_offsets = [node.offset for node in nodes]
        all_offsets = [frequency_to_timedelta(o) for o in all_offsets]
        all_offsets = all_offsets + [frequency_to_timedelta("0h")]
        minimum = min(all_offsets)
        maximum = max(all_offsets)

        freq = obj._context.frequency

        freq_seconds = round(freq.total_seconds())

        def round_down(dt):
            seconds = round(dt.total_seconds())
            return datetime.timedelta(seconds=seconds - (seconds % freq_seconds))

        minimum = round_down(minimum)

        def round_up(dt):
            seconds = round(dt.total_seconds())
            remainder = seconds % freq_seconds
            if remainder == 0:
                return datetime.timedelta(seconds=seconds)
            return datetime.timedelta(seconds=seconds + (freq_seconds - remainder))

        maximum = round_up(maximum)

        assert round(minimum.total_seconds()) % round(freq.total_seconds()) == 0
        assert round(maximum.total_seconds()) % round(freq.total_seconds()) == 0
        assert minimum <= maximum, f"Minimum offset {minimum} is greater than maximum offset {maximum}"

        dropped_samples = -seconds(minimum) / seconds(freq)
        dropped_samples += seconds(maximum) / seconds(freq)
        assert int(dropped_samples) == dropped_samples
        dropped_samples = int(dropped_samples)

        for node in nodes:
            node.set_min_max_offsets(minimum=minimum, maximum=maximum, dropped_samples=dropped_samples)

    # print('✅ created obj : ' , obj)
    return obj
