from typing import Literal

from pydantic import Field

from anemoi.utils.schemas import BaseModel


class SkipConnectionSchema(BaseModel):
    target_: Literal["anemoi.models.layers.residual.SkipConnection"] = Field(..., alias="_target_")


class TruncationMapperSchema(BaseModel):
    target_: Literal["anemoi.models.layers.residual.TruncationMapper"] = Field(..., alias="_target_")
