import datetime
import json
import warnings
from functools import cached_property

import numpy as np
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import ListConfig
from rich import print
from rich.console import Console
from rich.tree import Tree

from anemoi.datasets import open_dataset
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta


def resolve_reference(config):
    from omegaconf import OmegaConf

    config = OmegaConf.create(config)
    config = OmegaConf.to_container(config, resolve=True)
    return config


def seconds(td):
    return round(td.total_seconds())


class Context:
    def __init__(
        self,
        start=None,
        end=None,
        frequency=None,
        sources=None,
        data_config=None,
        offset=None,
        request=None,
        _list_of_leaves=None,
        _parent=None,
    ):
        # remove this
        if sources is None and data_config is not None:
            print("WARNING: 'data_config' is deprecated, use 'sources' instead.")
            sources = data_config
        del data_config
        #
        self._parent = _parent

        # if not specified, take the offset from the parent context
        # default offset is 0h
        if offset is None:
            offset = "0h"
        if isinstance(offset, str):
            offset = frequency_to_timedelta(offset)
        if _parent is not None:
            offset = _parent.offset + offset
        self.offset = offset

        # request from the parent always overrides the request in the current context
        if _parent is not None and _parent.request is not None:
            request = _parent.request
        if not isinstance(request, (type(None), list, str)):
            raise ValueError(f"Expected list or string for request, got {type(self.request)}: {self.request}.")
        self.request = request

        if _parent is not None:
            # TODO : refactor this nonsensical list of forwarding to _parent
            if not isinstance(_parent, Context):
                raise TypeError(f"Expected Context as parent, got {type(_parent)}: {_parent}")
            if start is None:
                start = _parent.start
            if end is None:
                end = _parent.end
            if frequency is None:
                frequency = _parent.frequency
            if sources is None:
                sources = _parent.sources
            if _list_of_leaves is None:
                _list_of_leaves = _parent._list_of_leaves

        self.start = start
        self.end = end
        self.frequency = frequency_to_timedelta(frequency)

        if _list_of_leaves is None:
            _list_of_leaves = []
        self._list_of_leaves = _list_of_leaves

        sources = resolve_reference(sources)
        self.sources = sources

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
        assert isinstance(
            self.request,
            (type(None), list, str),
        ), f"Expected list or string or None for request, got {type(self.request)}: {self.request}"

    def register_as_leaf(self, obj):
        # we should maybe use a set instead of a list, here?
        assert obj not in self._list_of_leaves, f"Object {obj} is already registered in {self._list_of_leaves}"
        self._list_of_leaves.append(obj)

    def __repr__(self):
        return f"Context(start={self.start}, end={self.end}, offset={self.offset})"


class VariablesList:
    def __init__(self, variables: list[str] | dict, data=None):
        if data is not None:
            warnings.warn(
                "Using 'data' argument is deprecated, use variables with '.' instead",
                DeprecationWarning,
            )
            if not isinstance(variables, (list, tuple)):
                raise ValueError(
                    f"Expected list or tuple for variables, got {type(variables)}: {variables}, data={data}",
                )
            self.lst = [f"{data}.{v}" for v in variables]
            return

        assert data is None, data

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

    @property
    def is_root(self):
        return self._parent is None

    def __init__(self, _context: Context, _parent):
        self._context = _context
        self._parent = _parent
        self._frequency = self._context.frequency

    def invite(self, visitor):
        visitor.visit(self)

    def mutate(self):
        return self

    def __len__(self):
        raise NotImplementedError(
            f"Length is not implemented for {self.__class__.__name__}. Please implement __len__ method.",
        )

    def latitudes(self, item: int):
        raise NotImplementedError()

    def longitudes(self, item: int):
        raise NotImplementedError()

    def timedeltas(self, item: int):
        raise NotImplementedError()

    @property
    def name_to_index(self, item: int):
        raise NotImplementedError(
            f"name_to_index is not implemented for {self.__class__.__name__}. Please implement name_to_index method."
        )

    def statistics(self, item: int):
        raise NotImplementedError(
            f"statistics is not implemented for {self.__class__.__name__}. Please implement statistics method."
        )

    def processors(self, item: int):
        raise NotImplementedError()

    def num_channels(self, item: int):
        raise NotImplementedError()

    def shape(self, item: int):
        raise NotImplementedError()

    @property
    def frequency(self):
        return self._frequency

    def __repr__(self):
        console = Console(record=True, width=120)
        tree = self._build_tree()
        with console.capture() as capture:
            console.print(tree)
        return capture.get()

    def _build_tree(self, label: str = None, **kwargs):
        raise NotImplementedError("Subclasses must implement _build_tree method in " + self.__class__.__name__)

    def _check_item(self, item: int):
        if not isinstance(item, (int, np.integer)):
            raise TypeError(f"Not implemented for non-integer indexing {type(item)}")

    def shuffle(self, *args, **kwargs):
        # TODO: remove doulg self, self
        return ShuffledSampleProvider(self._context, self, self, *args, **kwargs)


class ForwardSampleProvider(SampleProvider):
    def __init__(self, _context, _parent, sample: SampleProvider):
        super().__init__(_context, _parent)
        self._forward = sample

    def __len__(self):
        return len(self._forward)

    @property
    def name_to_index(self):
        return self._forward.name_to_index

    @property
    def statistics(self, item):
        return super().statistics(item)

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

    def __getitem__(self, item: int):
        print(f"Shuffling : requested {item}, provided {self.idx[item]}")
        return self._forward.__getitem__(self.idx[item])

    def _build_tree(self, prefix=""):
        tree = Tree(prefix + self.emoji + self.label + f" (seed={self.seed})")
        subtree = self._forward._build_tree()
        tree.add(subtree)
        return tree


class DictSampleProvider(SampleProvider):
    label = "dict"
    emoji = "📖"

    def __init__(self, _context: Context, _parent, dictionary: dict, with_attributes: bool = False):
        super().__init__(_context, _parent)
        self.with_attributes = with_attributes

        if not isinstance(dictionary, dict):
            raise TypeError(f"Expected dictionary, got {type(dictionary)}: {dictionary}")
        if len(dictionary) == 0:
            raise ValueError("Dictionary is empty, cannot create sample provider.")
        for k in dictionary:
            if not isinstance(k, str):
                raise ValueError(f"Keys in dictionary must be strings, got {type(k)}, {k}")

        def normalise_key(k):
            new_k = "".join([x.lower() if x.isalnum() else "_" for x in k])
            if k != new_k:
                warnings.warn(f"Normalising key '{k}' to '{new_k}'")
            return new_k

        dictionary = {normalise_key(k): v for k, v in dictionary.items()}

        for k, v in dictionary.items():
            if not isinstance(v, dict):
                raise ValueError(f"Expected dictionary for sample provider, got {type(v)}: {v}. ")
        self._samples = {k: sample_provider_factory(self._context, **v) for k, v in dictionary.items()}

    def __getattr__(self, key):
        if key in self._samples:
            return self._samples[key]
        raise AttributeError(f"{type(self).__name__} has no attribute '{key}'")

    def __getitem__(self, item):
        if item == 0:
            item = item + 1  # ✅✅ TODO provide the correct lenght
        return {k: v.__getitem__(item) for k, v in self._samples.items()}

    @property
    def name_to_index(self):
        return {k: v.name_to_index for k, v in self._samples.items()}

    @property
    def statistics(self):
        return {k: v.statistics for k, v in self._samples.items()}

    def _build_tree(self, prefix=""):
        tree = Tree(prefix + self.label)
        for k, v in self._samples.items():
            subtree = v._build_tree(prefix=f'"{k}" : ')
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
        self._forward = sample_provider_factory(new_context, **kwargs)

        # shift = self._offset_as_timedelta // self._forward.frequency

    def invite(self, visitor):
        super().invite(visitor)
        self._forward.invite(visitor)

    @property
    def name_to_index(self):
        return self._forward.name_to_index

    @property
    def statistics(self):
        return self._forward.statistics

    def __getitem__(self, item: int):
        return self._forward.__getitem__(item)

    def _build_tree(self, prefix: str = ""):
        tree = self._forward._build_tree()
        tree.label = tree.label + f" ({self.emoji} {self.values})"
        return tree

    def __len__(self):
        return len(self._forward)


class OffsetSampleProvider(_FilterSampleProvider):
    emoji = "⏱️"
    label = "Offset"
    keyword = "offset"


class RequestSampleProvider(_FilterSampleProvider):
    emoji = "🙏"
    label = "Request"
    keyword = "request"


class Dimension:
    def __init__(self, **raw):
        self.raw = raw

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self.raw.items())})"


class IterableDimension(Dimension):
    name = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        key = list(kwargs.keys())[0]
        assert key == self.name
        if isinstance(kwargs[key], ListConfig):
            kwargs[self.name] = list(kwargs[key])
        self.values = kwargs[key]
        self.check()

    def check(self):
        if not isinstance(self.values, (list, tuple)):
            raise ValueError(f"Not implemented for non-list values in {self.name}: {self.values}")

    def __repr__(self):
        return f"{self.name}({', '.join(map(str, self.values))})"


class OffsetDimension(IterableDimension):
    name = "offset"


class VariablesDimension(IterableDimension):
    name = "variables"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if isinstance(self.values, dict):
            self.values = [f"{k}.{v}" for k, vals in self.values.items() for v in vals]

    def check(self):
        pass


class RepeatDimension(IterableDimension):
    name = "repeat"


class DataDimension(Dimension):
    name = "data"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "name" in kwargs:
            self.name = kwargs["name"]


class SelectionDimension(Dimension):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        key = list(kwargs.keys())[0]
        assert key == self.name
        self.values = kwargs[key]


class EnsembleDimension(SelectionDimension):
    name = "ensembles"


class ValuesDimension(SelectionDimension):
    name = "values"


def dimension_factory(raw_dim):
    if isinstance(raw_dim, Dimension):
        return raw_dim
    if isinstance(raw_dim, DictConfig):
        raw_dim = dict(raw_dim)
    assert isinstance(raw_dim, dict), f"Expected dict, got {type(raw_dim)}: {raw_dim}"
    if "variables" in raw_dim:
        return VariablesDimension(**raw_dim)
    if "offset" in raw_dim:
        return OffsetDimension(**raw_dim)
    if "ensembles" in raw_dim:
        warnings.warn("Ensemble dimensions are not implemented yet, ignoring the config")
        return EnsembleDimension(**raw_dim)
    if "values" in raw_dim:
        warnings.warn("Values dimensions are not implemented yet, ignoring the config")
        return ValuesDimension(**raw_dim)
    if "repeat" in raw_dim:
        warnings.warn("repeat should only be used for testing.")
        return RepeatDimension(**raw_dim)
    if "lat_lon" in raw_dim:
        raise ValueError("'lat_lon' dimension is not supported'")
    return DataDimension(**raw_dim)


class TupleSampleProvider(SampleProvider):
    emoji = "🔗"
    label = "tuple"

    def __init__(self, _context: Context, _parent, tuple_: dict):
        super().__init__(_context, _parent)

        tuple_ = tuple_.copy()
        loops = tuple_.pop("loop")
        template = tuple_.pop("template")
        if tuple_:
            raise ValueError(f"Unexpected keys in tuple: {tuple_}")

        self.iterables = [dimension_factory(i) for i in loops]

        for i in self.iterables:
            if not isinstance(i, IterableDimension):
                raise ValueError(f"Expected iterable dimension in all loop dimensions, got {type(i)} for {i}")

        self.template = template

    def __len__(self):
        lenghts = [len(s) for s in self._samples]
        assert len(set(lenghts)) == 1, f"Samples in tuple have different lengths: {lenghts}. Cannot determine length."
        return lenghts[0]

    def invite(self, visitor):
        super().invite(visitor)
        for s in self._samples:
            s.invite(visitor)

    def mutate(self):
        if not self.iterables:
            return sample_provider_factory(self._context, **self.template)

        if len(self.iterables) == 1:
            iterable = self.iterables[0]
            self._samples = []
            for v in iterable.values:
                config = {iterable.name: v, "structure": self.template}
                print(f"Creating sample provider for {iterable.name} = {v} with config {config}")
                sample = sample_provider_factory(self._context, _parent=self._parent, **config)
                self._samples.append(sample)
            self._samples = tuple(self._samples)

            return self

        assert len(self.iterables) > 1
        *others, iterable = self.iterables
        new_config = {
            "tuple": {
                "loop": others,
                "template": {
                    "tuple": {
                        "loop": [iterable],
                        "template": self.template,
                    },
                },
            },
        }
        return sample_provider_factory(self._context, _parent=self._parent, **new_config)

    def __getitem__(self, item: int):
        def recurse(x):
            if isinstance(x, SampleProvider):
                return x.__getitem__(item)
            if isinstance(x, tuple):
                return tuple(recurse(elt) for elt in x)
            assert False, f"Unknown type {type(x)} : {x}"

        return recurse(self._samples)

    @property
    def name_to_index(self):
        return [s.name_to_index for s in self._samples]

    @property
    def statistics(self):
        return [s.statistics for s in self._samples]

    def _build_tree(self, prefix=""):
        tree = Tree(prefix + self.emoji + self.label + f" ({len(self._samples)} samples)")
        for s in self._samples:
            tree.add(s._build_tree())
        return tree


class TensorSampleProvider(SampleProvider):
    emoji = "🔢"
    label = "tensor"

    _tuple_sample_provider = None

    def __init__(self, _context: Context, _parent, tensor: dict):
        dims = tensor
        super().__init__(_context, _parent)

        self.dimensions = [dimension_factory(dim) for dim in dims]
        if len(dims) != len(list(set([dim.name for dim in self.dimensions]))):
            raise ValueError(f"Duplicate dimension names in tuple {dims}")

        self.order = [dim.name for dim in self.dimensions]  # keep the dim order to reshape

        template = [d for d in self.dimensions if isinstance(d, DataDimension)]
        if len(template) > 1:
            raise ValueError(f"Expected a single data dimension, got {template}")
        if len(template) == 0:
            template = [d for d in self.dimensions if isinstance(d, VariablesDimension)]
        if len(template) == 0:
            raise ValueError(f"Expected a data or variables dimension, got {self.dimensions}, nothing to point to data")
        self.template = template[0]

        self.loops = [d for d in self.dimensions if isinstance(d, IterableDimension) and d != self.template]

        config = {
            "tuple": {
                "loop": self.loops,
                "template": self.template.raw,
            },
        }
        self._tuple_sample_provider = sample_provider_factory(_context, _parent=self, **config)

    def __len__(self):
        return len(self._tuple_sample_provider)

    def invite(self, visitor):
        super().invite(visitor)
        self._tuple_sample_provider.invite(visitor)

    def __getitem__(self, item: int):
        data = self._tuple_sample_provider.__getitem__(item)

        def process_element(k, elt):
            if k == "data":
                elt = np.array(elt)
                self.transpose(elt)
                return elt
            else:
                return elt

        if isinstance(data, dict):
            return {k: process_element(k, v) for k, v in data.items()}

        return process_element("data", data)

    @property
    def name_to_index(self):
        sample = self._tuple_sample_provider
        x = sample.name_to_index
        return self._flatten(x)

    @property
    def statistics(self):
        sample = self._tuple_sample_provider
        x = sample.statistics
        return self._flatten(x)

    def _flatten(self, x):
        if isinstance(x, dict):
            return x
        assert len(set([str(_) for _ in x])) == 1, f"Expected a single name_to_index, got {x} for {self}"
        x = x[0]
        if isinstance(x, dict):
            return x
        raise NotImplementedError(
            f"name_to_index not implemented for tensor with more than two dimensions: {self.dimensions}"
        )

    def transpose(self, array):
        # Transpose the array to match the order of requested dimensions
        # TODO : clean up this logic, maybe use ... from einops
        array = np.array(array)

        if not isinstance(array, np.ndarray):
            return array

        dimensions = self.dimensions
        order = self.order.copy()

        if len(dimensions) < array.ndim:
            # if there are less dimensions than the array has, we need to add empty dimensions
            missing_dims = array.ndim - len(dimensions)
            dimensions = dimensions + [DataDimension(name=f"dim_{i}") for i in range(missing_dims)]
            order += [f"dim_{i}" for i in range(missing_dims)]

        assert (
            len(dimensions) == array.ndim
        ), f"Expected {len(self.dimensions)} dimensions, got {array.ndim} for {array}"
        assert len(order) == len(dimensions), f"Expected {len(self.dimensions)} order, got {len(order)} for {array}"

        current_order = [
            dim.name for dim in dimensions if isinstance(dim, IterableDimension) and not dim == self.template
        ] + [self.template.name]
        if len(current_order) != len(order):
            missing_dims = len(order) - len(current_order)
            current_order += [f"dim_{i}" for i in range(missing_dims)]

        assert len(current_order) == len(order), f"Current order {current_order} does not match requested order {order}"

        import einops

        return einops.rearrange(array, " ".join(current_order) + " -> " + " ".join(order))

    def _build_tree(self, prefix=""):
        tree = Tree(f"{prefix}{self.emoji} {self.label}")
        for i, d in enumerate(self.dimensions):
            if isinstance(d, IterableDimension):
                tree.add(f" dim {i} : {d}")
            elif isinstance(d, DataDimension):
                tree.add(self._tuple_sample_provider._build_tree(prefix=f"dim {i} : {d}"))
            else:
                tree.add(f'? dim {i} "{d.name}" = {d.raw}')

        if self._tuple_sample_provider is not None:
            subtree = self._tuple_sample_provider._build_tree(prefix="(debug) ")
            tree.add(subtree)
        return tree


class VariablesSampleProvider(SampleProvider):
    emoji = "⛁"  #  🆎
    # emoji = "🧩"
    label = "Variables"

    min_offset = None
    max_offset = None
    i_offset = None
    dropped_samples = None

    def __init__(self, _context: Context, _parent, variables: dict | list[str], data: str = None):
        super().__init__(_context, _parent)
        self.variables = VariablesList(variables, data=data)
        if len(self.variables.as_dict) > 1:
            raise ValueError(
                f"Expected a single group of variables, got {list(self.variables.as_dict.keys())} in {variables}",
            )

        dic = self.variables.as_dict
        self.group = list(dic.keys())[0]

        self.request = _context.request

        self.offset = frequency_to_timedelta(self._context.offset)
        self._context.register_as_leaf(self)

    @property
    def name_to_index(self):
        return self.data_handler.name_to_index

    @property
    def statistics(self):
        return self.data_handler.statistics

    def set_min_max_offsets(self, min_offset=None, max_offset=None, dropped_samples=None):
        self.min_offset = min_offset
        self.max_offset = max_offset
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
        return len(self.data_handler) - self.dropped_samples

    @cached_property
    def data_handler(self):
        simple_names = self.variables.as_dict[self.group]
        return DataHandler(self.group, variables=simple_names, sources=self._context.sources)

    def __getitem__(self, item):
        actual_item = item + self.i_offset

        if actual_item < 0 or actual_item >= len(self.data_handler):
            print("❌", self)
            msg = f"Item {item} ({actual_item}) is out of bounds with i_offset {self.i_offset}, lenght of the dataset is {len(self.data_handler)} and dropped_samples is {self.dropped_samples}."
            raise IndexError(msg)

        return self.data_handler._get(actual_item, request=self.request)

    def _build_tree(self, prefix: str = ""):
        def _(x):
            return frequency_to_string(x) if x else "0h"

        string_offset = _(self.offset)
        if not self.offset:
            string_offset = "0h"

        txt = f"{prefix}{self.emoji} {self.label} ("
        txt += f"{self.group}:{'/'.join(self.variables.as_list)}, offset={string_offset}"
        txt += ")"
        tree = Tree(txt)
        if self.min_offset is not None:
            tree.add(f"global_min_offset={_(self.min_offset)}")
        if self.max_offset is not None:
            tree.add(f"global_max_offset={_(self.max_offset)}")
        tree.add(f"lenght={len(self)}")
        if self.i_offset is not None:
            explain = f"({_(self.offset)} - {_(self.min_offset)})/{_(self.frequency)} = {_(self.actual_offset)}/{_(self.frequency)} = {self.i_offset}"
            tree.add(f"i -> i + {self.i_offset} because {explain}")
        return tree


class DataHandler:
    emoji = "D"
    label = "DataHandler"

    def __init__(self, group: str, variables: list = [], sources=None):
        self.group = group

        if sources is None:
            raise ValueError("For now, sources must be provided to DataHandler. For now.")

        if self.group not in sources:
            raise ValueError(
                f"Group '{self.group}' not found in sources: available groups are {list(sources.keys())}",
            )
        self.config = sources[self.group].copy()
        self.preprocessors = self.config.get("processors", {})
        self._configs = self.config.get("configs", {})

        variables = [f"{group}.{v}" for v in variables]
        self.variables = variables

        self.ds = open_dataset(dataset=self.config["dataset"], select=self.variables)

        self.frequency = frequency_to_timedelta(self.ds.frequency)
        self._statistics = self.ds.statistics[group]
        self._name_to_index = self.ds.name_to_index[group]

        if hasattr(self.ds, "shapes"):
            self._shape = self.ds.shapes[group]
        elif hasattr(self.ds, "shape"):
            if isinstance(self.ds.shape, dict):
                self._shape = self.ds.shape[group]
            elif isinstance(self.ds.shape, (list, tuple)):
                self._shape = self.ds.shape
            else:
                raise ValueError(f"Unexpected shape type {type(self.ds.shape)}: {self.ds.shape}")
        else:
            raise ValueError(f"Dataset {self.ds} does not have 'shape' or 'shapes' attribute.")

        # TODO:
        # this class is not efficient and should be refactored to make sure a chunk of data is loaded only once
        # this implies changing the VariablesSampleProvider class, compress the requests for data
        # and use pointers (views) to the data that will be loaded only once
        # but the interface using "request" may stay the same

    def invite(self, visitor):
        visitor.visit(self)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item: int):
        return self.ds[item][self.group]

    def latitudes(self, item=None):
        if item is None:
            return self.ds.latitudes[self.group]
        return self.ds[item].latitudes[self.group]

    def longitudes(self, item=None):
        if item is None:
            return self.ds.longitudes[self.group]
        return self.ds[item].longitudes[self.group]

    def timedeltas(self, item=None):
        if item is None:
            # not implemented ?
            return self.ds.timedeltas[self.group]
        return self.ds[item].timedeltas[self.group]

    def get_statistics(self, item=None):
        return self._statistics

    def get_shape(self, item=None):
        return self._shape

    def latitudes_longitudes(self, item=None):
        lats = self.latitudes(item)
        longs = self.longitudes(item)
        return np.array([lats, longs]).T

    def get_configs(self, item=None):
        return self._configs

    def get_name_to_index(self, item=None):
        return self._name_to_index

    def _get(self, item: int, request):
        assert isinstance(item, (type(None), int, np.integer)), f"Expected integer for item, got {type(item)}: {item}"
        assert isinstance(
            request,
            (type(None), str, list, tuple),
        ), f"Expected string or list for request, got {type(request)}: {request}"

        ACTIONS = {
            None: self.__getitem__,
            "data": self.__getitem__,
            "latitudes": self.latitudes,
            "longitudes": self.longitudes,
            "latitudes_longitudes": self.latitudes_longitudes,
            "timedeltas": self.timedeltas,
            "statistics": self.get_statistics,
            "shape": self.get_shape,
            "configs": self.get_configs,
            "name_to_index": self.get_name_to_index,
        }

        def do_action(r, item):
            action = ACTIONS.get(r)

            if action is not None:
                return action(item)

            if "." in r:
                rr, key = r.split(".")
                action = ACTIONS.get(rr)
                if action is None:
                    raise ValueError(
                        f"Unknown request '{r}' in {request}. Available requests are {list(ACTIONS.keys())}.",
                    )
                return action(item)[key]

            raise ValueError(f"Unknown request '{r}' in {request}. Available requests are {list(ACTIONS.keys())}.")

        if isinstance(request, (list, tuple)):
            dic = {}
            for r in request:
                dic[r] = do_action(r, item)
            return dic

        return do_action(request, item)

    @property
    def name_to_index(self):
        return self._name_to_index

    @property
    def statistics(self):
        return self._statistics

    def __repr__(self):
        return f"DataHandler {self.config['dataset']} @ {self.group} [{', '.join(self.variables)}]"


def sample_provider_factory(_context=None, **kwargs):
    kwargs = kwargs.copy()

    if "context" in kwargs:
        raise NotImplementedError(
            "The 'context' argument is deprecated, use directly start, end, frequency, sources instead",
        )

    if _context is None:
        _context = Context(
            sources=kwargs.pop("sources", None),
            start=kwargs.pop("start", None),
            end=kwargs.pop("end", None),
            frequency=kwargs.pop("frequency"),
        )

    if "_parent" not in kwargs:
        kwargs["_parent"] = None
        print(f"Building sample provider : {kwargs}")

    if "offset" in kwargs:
        obj = OffsetSampleProvider(_context, **kwargs)
    elif "request" in kwargs:
        obj = RequestSampleProvider(_context, **kwargs)
    elif "dictionary" in kwargs:
        obj = DictSampleProvider(_context, **kwargs)
    elif "tensor" in kwargs:
        obj = TensorSampleProvider(_context, **kwargs)
    elif "tuple" in kwargs:
        kwargs["tuple_"] = kwargs.pop("tuple")
        obj = TupleSampleProvider(_context, **kwargs)
    elif "variables" in kwargs:
        if isinstance(kwargs["variables"], ListConfig):
            kwargs["variables"] = list(kwargs["variables"])
        obj = VariablesSampleProvider(_context, **kwargs)
    elif "repeat" in kwargs:
        repeat = kwargs.pop("repeat")
        obj = sample_provider_factory(_context, **kwargs)
    elif "structure" in kwargs:
        if isinstance(kwargs["structure"], SampleProvider):
            return kwargs["structure"]  # not mutate here?: todo: think about it
        if isinstance(kwargs["structure"], DictConfig):
            kwargs["structure"] = dict(kwargs["structure"])
        if isinstance(kwargs["structure"], dict):
            obj = sample_provider_factory(_context, **kwargs["structure"])
        else:
            raise ValueError(
                f"Expected dictionary for 'structure', got {type(kwargs['structure'])}: {kwargs['structure']}",
            )
    else:
        assert False, f"Unknown sample type for kwargs {kwargs}"
    obj = obj.mutate()

    if obj.is_root:
        # set the min/max offsets for the root sample provider
        leaves = obj._context._list_of_leaves
        all_offsets = [leaf.offset for leaf in leaves]
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

        for leaf in leaves:
            leaf.set_min_max_offsets(min_offset=minimum, max_offset=maximum, dropped_samples=dropped_samples)

    check_sample_provider(obj)

    return obj


def check_sample_provider(obj):
    # check there is one root, and only one
    class FindRoot:
        def __init__(self):
            self.roots = []

        def visit(self, sp):
            if sp.is_root:
                self.roots.append(sp)
            return self

    roots = FindRoot().visit(obj).roots
    if len(roots) > 1:
        print(f"Found roots: {roots}")
        for r in roots:
            print(r._build_tree())
        raise ValueError(f"Multiple root sample providers found: {roots}. Please ensure only one root is defined.")
    if len(roots) == 0 and obj.is_root:
        raise ValueError("No root sample provider found. Please ensure a root sample provider is defined.")
    if len(roots) == 1 and not obj.is_root:
        raise ValueError(
            f"Sample provider {obj} is not a root, but it should be. Please ensure the root sample provider is defined.",
        )


# TEST ---------------------------------
if __name__ == "__main__":
    yaml_str = """
sources:
  training:
    era5:
      dataset:
        dataset: aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8
        set_group: era5
      # processors: ...
      optimisations: use_tensors
      user_key: mine
    snow:
      dataset: observations-testing-2018-2018-6h-v0
    metop_a:
      dataset: observations-testing-2018-2018-6h-v0
      configs:
        user_key_1:
            "metop_a.scatss_1": "foo"
            "metop_a.scatss_2": ["bar"]
        user_key_2:
            "metop_a.scatss_1": "foo"
            "metop_a.scatss_2": ["bar"]
        normaliser:
            "metop_a.scatss_1": "mean-std"
            "scatss_2": "min-max"
            "scatss_3": {"name": "custom-normaliser", "theta": 0.5, "rho": 0.1}



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

        ex_request_1:
          tensor:
            - variables: ["metop_a.scatss_1", "metop_a.scatss_2"]
              request: data
              # this is the default

        ex_request_2:
          tensor:
            - variables: ["metop_a.scatss_1", "metop_a.scatss_2"]
              request: latitudes

        ex_request_3:
          tensor:
            - variables: ["metop_a.scatss_1", "metop_a.scatss_2"]
              request: [data, latitudes]

        ex_request_4:
          request: [data, latitudes_longitudes, timedeltas]
          tuple:
            loop:
              - offset: [-6h, 0h]
            template:
              variables: ["metop_a.scatss_1", "metop_a.scatss_2"]

        ex_request_4:
          request: [statistics, shape, name_to_index]
          tuple:
            loop:
              - offset: [-6h, 0h]
            template:
              variables: ["metop_a.scatss_1", "metop_a.scatss_2"]


        #test_offset4:
        #  offset: "-6h"
        #  structure:
        #    offset: "-6h"
        #    structure:
        #      variables: ["metop_a.scatss_1", "metop_a.scatss_2"]
        #
        #test_request1:
        #  request: [data, shape]
        #  structure:
        #      variables: ["metop_a.scatss_1", "metop_a.scatss_2"]
        #      request: [data, latitudes_longitudes, timedeltas]
        #
        #test_request2:
        #  variables: ["metop_a.scatss_1", "metop_a.scatss_2"]
        #  request: configs
        #
        #test_request3:
        #  variables: ["metop_a.scatss_1", "metop_a.scatss_2"]
        #  request: configs.normaliser

#        et_implemented:
#          tuple:
#            - offset: ["-12h", "-6h"]
#            - ensembles: 1
#            - variables: ["metop_a.scatss_1", "metop_a.scatss_2", "snow.sdepth_0"]
#            - dictionary:
#                key:
#                  variables: ["metop_a.scatss_1", "metop_a.scatss_2", "snow.sdepth_0"]

"""

    import sys

    if len(sys.argv) > 1:
        import yaml

        path = yaml.safe_load(sys.argv[1])
        with open(path, "r") as f:
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
        # sources=CONFIG["data"],
        sources=sources_config,
        start=None,
        end=None,
        frequency="6h",
    )
    for key, config in sample_config["dictionary"].items():
        print(f"[yellow]- {key} : building sample_provider[/yellow]")
        print(yaml.dump(config, indent=2, sort_keys=False))
        s = sample_provider_factory(**training_context, **config)
        print(s)
        print(len(s))
        print("----------------------------")

    print("✅✅  --------")
    for key, config in sample_config["dictionary"].items():
        print(f"[yellow]- {key} : getting data [/yellow]")
        print(yaml.dump(config, indent=2, sort_keys=False))
        s = sample_provider_factory(**training_context, **config)
        print(s)
        name_to_index = s.name_to_index
        print(f"name_to_index = {name_to_index}")
        statistitics = s.statistics
        print(f"statistics = {statistitics}")
        print("sp[1] = ", show_json(s[1]))
        print()
