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
        **kwargs,
    ):
        if "data_config" in kwargs:
            raise ValueError("The 'data_config' argument is deprecated. Use 'sources' instead.")
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {kwargs}")
        del kwargs

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
    min_offset = None
    max_offset = None

    def __init__(self, _context: Context, _parent):
        _context.register(self)
        self._context = _context
        self._parent = _parent
        self._frequency = _context.frequency
        self.offset = frequency_to_timedelta(_context.offset)

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
        raise NotImplementedError()

    def longitudes(self, item: int):
        raise NotImplementedError()

    def timedeltas(self, item: int):
        raise NotImplementedError()

    @property
    def name_to_index(self, item: int):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    @property
    def statistics(self):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    @property
    def dataspecs(self):
        return self._path(prefix="")

    def _path(self, prefix):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    @property
    def extra(self):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    @property
    def normaliser(self):
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

    def get_native(self, item: int):
        dic = self[item]
        if "dataspecs" not in dic:
            dic["dataspecs"] = self.dataspecs
        return dic

    def get_obj(self, item):
        raw = self.get_native(item)
        if "dataspecs" not in raw:
            raw["dataspecs"] = self.dataspecs
        return structure_factory(**raw)

    def get_things_that_do_not_depend_on_the_index(self):
        """Returns a structure that does not depend on the index."""
        return structure_factory(
            name_to_index=self.name_to_index,
            statistics=self.statistics,
            dataspecs=self.dataspecs,
            extra=self.extra,
            normaliser=self.normaliser,
            imputer=self.imputer,
        )

    def create_structure_from_batch(self, batch):
        """Creates a structure from a batch of data."""
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict for batch, got {type(batch)}: {batch}")
        return structure_factory(
            dataspecs=self.dataspecs,
            **batch,
        )

    @property
    def shape(self):
        raise NotImplementedError()

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
    def name_to_index(self):
        return self._forward.name_to_index

    @property
    def statistics(self):
        return self._forward.statistics

    def _path(self, prefix):
        return self._forward._path(prefix)

    @property
    def extra(self):
        return self._forward.extra

    @property
    def normaliser(self):
        return self._forward.normaliser

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

    def __getitem__(self, item: int):
        print(f"Shuffling : requested {item}, provided {self.idx[item]}")
        return self._forward.__getitem__(self.idx[item])

    def tree(self, prefix=""):
        tree = Tree(prefix + self.emoji + self.label + f" (seed={self.seed})")
        subtree = self._forward.tree()
        tree.add(subtree)
        return tree


class DictSampleProvider(SampleProvider):
    label = "dict"
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

        def normalise_key(k):
            new_k = "".join([x.lower() if x.isalnum() else "_" for x in k])
            if k != new_k:
                warnings.warn(f"Normalising key '{k}' to '{new_k}'")
            return new_k

        dictionary = {normalise_key(k): v for k, v in dictionary.items()}

        for k, v in dictionary.items():
            if not isinstance(v, dict):
                raise ValueError(f"Expected dictionary for sample provider, got {type(v)}: {v}. ")

        self._samples = {k: sample_provider_factory(_context, **v) for k, v in dictionary.items()}

    def __getattr__(self, key):
        if key in self._samples:
            return self._samples[key]
        raise AttributeError(f"{type(self).__name__} has no attribute '{key}'")

    def __getitem__(self, item):
        dict_of_dict = {k: s[item] for k, s in self._samples.items()}
        first = dict_of_dict[list(dict_of_dict.keys())[0]]
        keys = list(first.keys())
        dic = {}
        for key in keys:
            dic[key] = {k: dict_of_dict[k][key] for k in self._samples.keys()}
        return dic

    @property
    def name_to_index(self):
        return {k: v.name_to_index for k, v in self._samples.items()}

    @property
    def statistics(self):
        return {k: v.statistics for k, v in self._samples.items()}

    def _path(self, prefix):
        first_dot = "" if prefix == "" else "."
        return {k: v._path(f"{prefix}{first_dot}{k}") for k, v in self._samples.items()}

    @property
    def extra(self):
        return {k: v.extra for k, v in self._samples.items()}

    @property
    def normaliser(self):
        return {k: v.normaliser for k, v in self._samples.items()}

    @property
    def imputer(self):
        return {k: v.imputer for k, v in self._samples.items()}

    @property
    def processors(self):
        return {k: v.processors for k, v in self._samples.items()}

    @property
    def shape(self):
        return {k: v.shape for k, v in self._samples.items()}

    def tree(self, prefix=""):
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
        self._forward = sample_provider_factory(new_context, **kwargs)

        # shift = self._offset_as_timedelta // self._forward.frequency

    @property
    def name_to_index(self):
        return self._forward.name_to_index

    @property
    def statistics(self):
        return self._forward.statistics

    def _path(self, prefix):
        return self._forward._path(prefix)

    @property
    def extra(self):
        return self._forward.extra

    @property
    def normaliser(self):
        return self._forward.normaliser

    @property
    def imputer(self):
        return self._forward.imputer

    @property
    def processors(self):
        return self._forward.processors

    @property
    def shape(self):
        return self._forward.shape

    def __getitem__(self, item: int):
        return self._forward.__getitem__(item)

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

    def __len__(self):
        return len(self.values)


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

    def __len__(self):
        return "undefined length, not iterable"


class SelectionDimension(Dimension):
    # not really used, yet
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

    def mutate(self):
        if not self.iterables:
            return sample_provider_factory(self._context, **self.template)

        if len(self.iterables) == 1:
            iterable = self.iterables[0]
            self._samples = []
            for v in iterable.values:
                config = {iterable.name: v, "structure": self.template}
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
                tuple_of_dicts = tuple(recurse(elt) for elt in x)
                first = tuple_of_dicts[0]
                keys = list(first.keys())
                dic = {}
                for key in keys:
                    # (tuple_of_dicts[key][i] for i in range(len(tuple_of_dicts))) for key in keys()}
                    _tuple = tuple(tuple_of_dicts[i][key] for i in range(len(tuple_of_dicts)))
                    dic[key] = _tuple
                return dic
            assert False, f"Unknown type {type(x)} : {x}"

        return recurse(self._samples)

    @property
    def name_to_index(self):
        return [s.name_to_index for s in self._samples]

    @property
    def statistics(self):
        return [s.statistics for s in self._samples]

    def _path(self, prefix):
        return [s._path(f"{prefix}.{i}") for i, s in enumerate(self._samples)]

    @property
    def extra(self):
        return [s.extra for s in self._samples]

    @property
    def normaliser(self):
        return [s.normaliser for s in self._samples]

    @property
    def imputer(self):
        return [s.imputer for s in self._samples]

    @property
    def processors(self):
        return [s.processors for s in self._samples]
    
    @property
    def shape(self):
        return [s.shape for s in self._samples]

    def tree(self, prefix=""):
        txt = prefix + self.emoji + self.label + f" ({len(self._samples)} samples)"
        tree = Tree(txt)
        for s in self._samples:
            tree.add(s.tree())
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

        self._template_sample_provider = sample_provider_factory(_context, _parent=self, **self.template.raw)

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

    def __getitem__(self, item: int):
        data = self._tuple_sample_provider.__getitem__(item)

        def process_element(k, elt):
            if k == "data":
                elt_ = np.array(elt)
                elt = self.transpose(elt_)
                return elt
            else:
                return elt

        assert "data" in data, f"Expected 'data' key in {data}, got {data}"
        return {k: process_element(k, v) for k, v in data.items()}

    @property
    def name_to_index(self):
        return dict(variables=self._template_sample_provider.name_to_index)

    @property
    def statistics(self):
        return dict(variables=self._template_sample_provider.statistics)

    def _path(self, prefix):
        return f"{prefix}.tensor"

    @property
    def extra(self):
        return self._template_sample_provider.extra

    @property
    def normaliser(self):
        return self._template_sample_provider.normaliser

    @property
    def imputer(self):
        return self._template_sample_provider.imputer

    @property
    def processors(self):
        return self._template_sample_provider.processors

    @property
    def shape(self):
        # todo read from data handler and have 'dynamic' shape
        return self[0].shape

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

    def tree(self, prefix=""):
        tree = Tree(f"{prefix}{self.emoji} {self.label}")
        for i, d in enumerate(self.dimensions):
            if isinstance(d, IterableDimension):
                tree.add(f" dim {i} : {d}")
            elif isinstance(d, DataDimension):
                tree.add(self._tuple_sample_provider.tree(prefix=f"dim {i} : {d}"))
            else:
                tree.add(f'? dim {i} "{d.name}" = {d.raw}')

        if self._tuple_sample_provider is not None:
            subtree = self._tuple_sample_provider.tree(prefix="(debug) ")
            tree.add(subtree)
        return tree


class VariablesSampleProvider(SampleProvider):
    emoji = "⛁"  #  🆎
    # emoji = "🧩"
    label = "Variables"

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

    @property
    def name_to_index(self):
        return self.data_handler.name_to_index

    @property
    def statistics(self):
        return self.data_handler.statistics

    def _path(self, prefix):
        assert False, "Should not be called, Tensor should be a parent and handle the request"

    @property
    def extra(self):
        return self.data_handler.extra

    @property
    def normaliser(self):
        return self.data_handler.normaliser

    @property
    def imputer(self):
        return self.data_handler.imputer

    @property
    def processors(self):
        return self.data_handler.processors

    @property
    def shape(self):
        return self.data_handler.shape

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

        return self.data_handler._get(actual_item, request=["data", "latitudes", "longitudes", "timedeltas"])

    def tree(self, prefix: str = ""):
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
        self._imputer_config = self.config.get("imputer", {})
        self._normaliser_config = self.config.get("normaliser", {})
        self._extra_config = self.config.get("extra", {})

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
            timedeltas = self.ds.timedeltas[self.group]
        else:
            timedeltas = self.ds[item].timedeltas[self.group]
        return timedeltas.astype("timedelta64[s]").astype(int)

    def get_statistics(self, item=None):
        warnings.warn(
            "using statistics is here deprecated, use sample_provider.statistics instead",
            DeprecationWarning,
        )
        return self._statistics

    def get_shape(self, item=None):
        return self._shape

    def latitudes_longitudes(self, item=None):
        lats = self.latitudes(item)
        longs = self.longitudes(item)
        return np.array([lats, longs]).T

    def get_name_to_index(self, item=None):
        warnings.warn(
            "using name_to_index here is deprecated, use sample_provider.name_to_index instead",
            DeprecationWarning,
        )
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
            "shape": self.get_shape,
            "name_to_index": self.get_name_to_index,  # moved to sample_provider.name_to_index
            "statistics": self.get_statistics,  # moved to sample_provider.statistics
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

    @property
    def imputer(self):
        return self._imputer_config

    @property
    def normaliser(self):
        return self._normaliser_config
    
    @property
    def processors(self):
        return dict(normaliser=self._normaliser_config, imputer=self._imputer_config)
    
    @property
    def extra(self):
        return self._extra_config

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
        # print(f"Building sample provider : {kwargs}")

    if "offset" in kwargs:
        obj = OffsetSampleProvider(_context, **kwargs)
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

    return obj


class StructureMixin:
    def __repr__(self, **kwargs):
        console = Console(record=True, width=120)
        tree = self.tree(**kwargs)
        with console.capture() as capture:
            console.print(tree)
        return capture.get()


class TupleStructure(StructureMixin, tuple):
    # beware, inheriting from tuple, do not use __init__ method

    def tree(self, prefix="", **kwargs):
        tree = Tree(prefix + "🔗")
        for v in self:
            tree.add(v.tree(**kwargs))
        return tree

    def apply(self, func):
        return TupleStructure([x.apply(func) for x in self])

    def __call__(self, structure, **kwargs):
        assert isinstance(structure, TupleStructure), f"Expected TupleStructure, got {type(structure)}: {structure}"
        return TupleStructure(func(elt, **kwargs) for func, elt in zip(self, structure))

    def _as_native(self):
        return tuple(x._as_native() for x in self)


class DictStructure(StructureMixin, dict):
    def __init__(self, content):
        super().__init__(content)
        for k in self.keys():
            if not hasattr(self, k):
                setattr(self, k, self[k])

    def tree(self, prefix="", **kwargs):
        tree = Tree(prefix + "📖")
        for k, v in self.items():
            tree.add(v.tree(prefix=f"{k} : ", **kwargs))
        return tree

    def apply(self, func):
        return DictStructure({k: v.apply(func) for k, v in self.items()})

    def __call__(self, structure, **kwargs):
        assert isinstance(structure, DictStructure), f"Expected DictStructure, got {type(structure)}: {structure}"
        assert set(self.keys()) == set(structure.keys()), f"Keys do not match: {self.keys()} vs {structure.keys()}"
        return DictStructure({k: self[k](structure[k], **kwargs) for k in self.keys()})

    def _as_native(self):
        return {k: v._as_native() for k, v in self.items()}


class LeafStructure(StructureMixin):
    def __init__(self, **content):
        for k, v in content.items():
            setattr(self, k, v)
        self._names = list(content.keys())

    def tree(self, prefix="", verbose=False, **kwargs):
        tree = Tree(f"{prefix}  📦 {', '.join(self._names)}")
        if hasattr(self, "shape"):
            tree.add(f"Shape: {self.shape}")
        if hasattr(self, "statistics"):
            tree.add(f"Statistics: {str(self.statistics)[:50]}")
        if hasattr(self, "latitudes"):
            try:
                txt = f"[{np.min(self.latitudes):.2f}, {np.max(self.latitudes):.2f}]"
            except ValueError:
                txt = f"[no np.min/np.max]"
            tree.add(f"Latitudes: [{txt}")
        if hasattr(self, "longitudes"):
            try:
                txt = f"[{np.min(self.longitudes):.2f}, {np.max(self.longitudes):.2f}]"
            except ValueError:
                txt = f"[no np.min/np.max]"
            tree.add(f"Longitudes: [{txt}]")
        if hasattr(self, "timedeltas"):
            try:
                minimum = np.min(self.timedeltas)
                maximum = np.max(self.timedeltas)
                minimum = int(minimum)
                maximum = int(maximum)
                minimum = datetime.timedelta(seconds=minimum)
                maximum = datetime.timedelta(seconds=maximum)
                minimum = frequency_to_string(minimum)
                maximum = frequency_to_string(maximum)
            except ValueError:
                minimum = "no np.min"
                maximum = "no np.max"
            tree.add(f"Timedeltas: [{minimum},{maximum}]")
        if hasattr(self, "data"):
            tree.add(f"Data: array of shape {self.data.shape} with mean {np.nanmean(self.data):.2f}")
        if hasattr(self, "dataspecs") and verbose:
            tree.add(f"dataspecs: {self.dataspecs}")
        return tree

    def apply(self, func):
        new = func(**{k: getattr(self, k) for k in self._names})
        name = func.__name__
        return LeafStructure(**{"dataspecs": self.dataspecs, name: new})

    def __call__(self, structure, function=None, input=None, result=None, **kwargs):
        assert isinstance(structure, LeafStructure), f"Expected LeafStructure, got {type(structure)}: {structure}"

        if not function:
            if len(self._names) != 2:
                raise NotImplementedError(f"Expected two names in LeafStructure, got {self._names}. ")
            for function in self._names:
                if function == "dataspecs":
                    continue
                break

        input_name = input or "data"
        result_name = result or input_name

        func = getattr(self, function)

        if not callable(func):
            raise ValueError(f"Expected a callable function in {self.dataspecs}, got {type(func)}: {func}")

        x = getattr(structure, input_name)
        y = func(x)

        return LeafStructure(**{"dataspecs": structure.dataspecs, result_name: y})

    def _as_native(self, key=None):
        if key is not None:
            return getattr(self, key, None)
        return {k: getattr(self, k) for k in self._names}


def structure_factory(**content):
    check_structure(**content)
    dataspecs = content["dataspecs"]

    if isinstance(dataspecs, str) and dataspecs.endswith(".tensor"):
        return LeafStructure(**content)

    if isinstance(dataspecs, (list, tuple)):
        lst = []
        for i in range(len(dataspecs)):
            lst.append(structure_factory(**{key: content[key][i] for key in content.keys()}))
        return TupleStructure(lst)

    assert isinstance(dataspecs, dict), f"Expected dicts"
    dic = {}
    for k in dataspecs.keys():
        dic[k] = structure_factory(**{key: content[key][k] for key in content.keys()})
    return DictStructure(dic)


def check_structure(**content):
    assert "dataspecs" in content, f"Missing 'dataspecs' in content"

    dataspecs = content["dataspecs"]
    for v in content.values():
        if isinstance(dataspecs, str) and dataspecs.endswith(".tensor"):
            continue

        if isinstance(dataspecs, dict):
            assert isinstance(
                v, dict
            ), f"Expected all values to be dict, got {type(v)} != {type(dataspecs)} whith {v} and {dataspecs}"
            assert set(v.keys()) == set(dataspecs.keys()), f"Expected the same keys, got {list(v.keys())} vs. {list(dataspecs.keys())}"

        if isinstance(dataspecs, (list, tuple)):
            assert isinstance(
                v, (list, tuple)
            ), f"Expected all values to be lists or tuples, got {type(v)} != {type(dataspecs)} whith {v} and {dataspecs}"
            assert len(v) == len(dataspecs), f"Expected the same length as first, got ✅{v}✅ vs ❌{dataspecs}❌"


# TEST ---------------------------------
if __name__ == "__main__":
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
    if True:
        # if False:
        # for key, config in sample_config["dictionary"].items():
        #     print(f"[yellow]- {key} : building sample_provider[/yellow]")
        #     print(yaml.dump(config, indent=2, sort_keys=False))
        #     s = sample_provider_factory(**training_context, **config)
        #     print(s)
        #     print("----------------------------")

        print("✅✅  --------")
        for key, config in sample_config["dictionary"].items():
            print(f"[yellow]- {key} : getting data [/yellow]")
            print(yaml.dump(config, indent=2, sort_keys=False))
            s = sample_provider_factory(**training_context, **config)
            print(s)
            print("length : ", len(s))
            name_to_index = s.name_to_index
            print(f"name_to_index = {name_to_index}")
            statistitics = s.statistics
            print(f"statistics = {statistitics}")
            print("sp[1] = ", show_json(s[1]))

    print("............................")

    i = 1

    config = """dictionary:
        fields:
          tensor:
            - variables: ["era5.2t", "era5.10u", "era5.10v"]
            - offset: ["-6h"]
        other_fields:
          tensor:
            - offset: ["-6h", "+6h"]
            - variables: ["era5.2t", "era5.10u"]
        observations:
          tuple:
            loop:
              - offset: ["-6h", "0h", "+6h"]
            template:
              tensor:
                - variables: ["metop_a.scatss_1", "metop_a.scatss_2"]
    """
    config = yaml.safe_load(config)
    sp = sample_provider_factory(**training_context, **config)

    content = {
        "name_to_index": sp.name_to_index,
        "statistics": sp.statistics,
        "extra": sp.extra,
        "dataspecs": sp.dataspecs,
        "normaliser": sp.normaliser,
        # "shape": sp.shape,
        # "data_spec": sp.data_spec,
        **sp[1],
    }
    print("info_from_sample_provider", content)

    data = sp[i]
    print("Native types data : ")
    for k, v in data.items():
        print(f"data['{k}'] = {show_json(v)}")

    obj = structure_factory(**content)
    print("Data as object :")
    print(obj)
    print(f"{obj.fields.name_to_index=}")
    print(f"{obj.fields.normaliser=}")
    print(f"{obj.fields.extra=}")
    print(f"{obj.fields.statistics=}")
    print(f"{obj.fields.dataspecs=}")
    print(f"{obj.observations[0].statistics=}")
    print(f"{obj.observations[0].data=}")
    print(f"{obj.observations[0].dataspecs=}")

    def my_function(name_to_index, statistics, normaliser, **kwargs):
        return f"Normalisers build from: {name_to_index=}, {statistics=}, {normaliser=}"

    result = obj.apply(my_function)
    print(result)
    print()
    print(f"{result.observations[0].my_function=}")
    print(f"{result.fields.my_function=}")
    print()
    print(f"{result.fields=}")
    print(f"{result.fields._as_native()}")
    print(f"{result.fields._as_native('my_function')=}")

    print(f"{str(sp.get_native(2))[:1000]=}")
    print(f"{sp.get_obj(2)=}")
    # print(sp.get_obj(2).__repr__(verbose=True))

    def double_function(normaliser, **kwargs):
        # print(normaliser)
        # return Normaliser(config = normaliser)
        return lambda x: x * 2

    function_structure = obj.apply(double_function)
    print("Function structure:")
    print(function_structure)

    result = function_structure(obj)
    print("Result of applying the function to the structure:")
    print(result)
