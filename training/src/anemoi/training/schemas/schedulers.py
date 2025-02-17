# (C) Copyright 2024-2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING
from typing import Literal
from typing import Union

from pydantic import Field
from pydantic import PositiveInt

from .utils import BaseModel

from anemoi.training.schedulers.schedulers import VALID_INCREMENT_TYPE
from anemoi.training.schedulers.schedulers import VALID_STEP_TYPES


class ImplementedRollout(str, Enum):
    static = "anemoi.training.schedulers.rollout.Static"
    positional = "anemoi.training.schedulers.rollout.indexed.PositionalIndexed"
    positional_epoch = "anemoi.training.schedulers.rollout.indexed.EpochPositionalIndexed"
    positional_step = "anemoi.training.schedulers.rollout.indexed.StepPositionalIndexed"
    lookup = "anemoi.training.schedulers.rollout.indexed.Lookup"
    lookup_epoch = "anemoi.training.schedulers.rollout.indexed.EpochLookup"
    lookup_step = "anemoi.training.schedulers.rollout.indexed.StepLookup"
    random_list = "anemoi.training.schedulers.rollout.randomise.RandomList"
    random_range = "anemoi.training.schedulers.rollout.randomise.RandomRange"
    increasing_random = "anemoi.training.schedulers.rollout.randomise.IncreasingRandom"
    increasing_random_epoch = "anemoi.training.schedulers.rollout.randomise.EpochIncreasingRandom"
    increasing_random_step = "anemoi.training.schedulers.rollout.randomise.StepIncreasingRandom"
    stepped = "anemoi.training.schedulers.rollout.stepped.Stepped"
    stepped_epoch = "anemoi.training.schedulers.rollout.stepped.EpochStepped"
    stepped_step = "anemoi.training.schedulers.rollout.stepped.StepStepped"


class BaseRolloutSchema(BaseModel):
    target_: ImplementedRollout = Field(..., alias="_target_")
    "Target rollout schema from anemoi.training.schedulers.rollout."


class StaticRolloutSchema(BaseRolloutSchema):
    """Static rollout configuration."""

    rollout_value: PositiveInt = Field(example=1)
    "Static rollout value."


class PositionalIndexedSchema(BaseRolloutSchema):
    """Positional indexed based rollout."""

    rollouts: list[PositiveInt] = Field(example=[1, 2, 3, 4])
    "List of rollout values."
    num_times_per_element: PositiveInt = Field(example=1)
    "Number of times to remain at an element."
    step_type: VALID_STEP_TYPES = Field(example="epoch")
    "Type of step, either 'epoch' or 'step'."


class EpochPositionalIndexedSchema(PositionalIndexedSchema):
    """Positional indexed based rollout based on epoch."""

    step_type: Literal["epoch"] = "epoch"


class StepPositionalIndexedSchema(PositionalIndexedSchema):
    """Positional indexed based rollout based on step."""

    step_type: Literal["step"] = "step"


class LookupSchema(BaseRolloutSchema):
    """Lookup dictionary based rollout."""

    table: dict[PositiveInt, PositiveInt] = Field(example={1: 1, 2: 2, 3: 3})
    "Lookup dictionary to from `step_type` to rollout value."
    step_type: VALID_STEP_TYPES = Field(example="epoch")
    "Type of step, either 'epoch' or 'step'."


class EpochLookupSchema(LookupSchema):
    """Lookup dictionary based rollout based on epoch."""

    step_type: Literal["epoch"] = "epoch"


class StepLookupSchema(LookupSchema):
    """Lookup dictionary based rollout based on step."""

    step_type: Literal["step"] = "step"


class RandomListSchema(BaseRolloutSchema):
    """Random list based rollout."""

    rollouts: list[PositiveInt] = Field(example=[1, 2, 3, 4])
    "List of rollout values."


class RandomRangeSchema(BaseRolloutSchema):
    """Random sample from range based rollout."""

    minimum: PositiveInt = Field(example=1)
    "Minimum rollout value."
    maximum: PositiveInt = Field(example=10)
    "Maximum rollout value."
    step: PositiveInt = Field(example=1)
    "Step of range."


class IncreasingRandomSchema(BaseRolloutSchema):
    """Increasing random range sample based rollout."""

    minimum: PositiveInt = Field(example=1)
    "Minimum rollout value."
    maximum: int = Field(example=10)
    "Maximum rollout value."
    range_step: PositiveInt = Field(example=1)
    "Step of range."
    every_n: PositiveInt = Field(example=5)
    "Number of steps or epochs to step the rollout value."
    increment: VALID_INCREMENT_TYPE = Field(1, example=1)
    "Value to increment the rollout by."
    step_type: VALID_STEP_TYPES = Field(example="epoch")
    "Type of step, either 'epoch' or 'step'."


class EpochIncreasingRandomSchema(IncreasingRandomSchema):
    """Increasing random range sample based rollout on epoch."""

    step_type: Literal["epoch"] = "epoch"
    every_n: PositiveInt = Field(example=5, alias="every_n_epochs")


class StepIncreasingRandomSchema(IncreasingRandomSchema):
    """Increasing random range sample based rollout on step."""

    step_type: Literal["step"] = "step"
    every_n: PositiveInt = Field(example=5, alias="every_n_steps")


class SteppedSchema(BaseRolloutSchema):
    """Increasing random range sample based rollout."""

    minimum: PositiveInt = Field(example=1)
    "Minimum rollout value."
    maximum: int = Field(example=10)
    "Maximum rollout value."
    every_n: PositiveInt = Field(example=5)
    "Number of steps or epochs to step the rollout value."
    increment: VALID_INCREMENT_TYPE = Field(1, example=1)
    "Value to increment the rollout by."
    step_type: VALID_STEP_TYPES = Field(example="epoch")
    "Type of step, either 'epoch' or 'step'."


class EpochSteppedSchema(SteppedSchema):
    """Increasing random range sample based rollout on epoch."""

    step_type: Literal["epoch"] = "epoch"
    every_n: PositiveInt = Field(example=5, alias="every_n_epochs")


class StepSteppedSchema(SteppedSchema):
    """Increasing random range sample based rollout on step."""

    step_type: Literal["step"] = "step"
    every_n: PositiveInt = Field(example=5, alias="every_n_steps")


RolloutSchemas = Union[
    StaticRolloutSchema,
    PositionalIndexedSchema,
    EpochPositionalIndexedSchema,
    StepPositionalIndexedSchema,
    LookupSchema,
    EpochLookupSchema,
    StepLookupSchema,
    RandomListSchema,
    RandomRangeSchema,
    IncreasingRandomSchema,
    EpochIncreasingRandomSchema,
    StepIncreasingRandomSchema,
    SteppedSchema,
    EpochSteppedSchema,
    StepSteppedSchema,
]
"Union of all rollout schemas."
