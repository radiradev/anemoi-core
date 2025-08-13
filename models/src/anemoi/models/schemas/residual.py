from typing import Literal

from pydantic import Field

from anemoi.utils.schemas import BaseModel


class SkipConnectionSchema(BaseModel):
    target_: Literal["anemoi.models.layers.residual.SkipConnection"] = Field(
        default="anemoi.models.layers.residual.SkipConnection", alias="_target_"
    )


class NoConnectionSchema(BaseModel):
    target_: Literal["anemoi.models.layers.residual.NoConnection"] = Field(..., alias="_target_")


class TruncatedConnectionSchema(BaseModel):
    target_: Literal["anemoi.models.layers.residual.TruncatedConnectionSchema"] = Field(..., alias="_target_")
    data_nodes: str = Field(..., description="Name of data nodes in the graph.")
    truncation_nodes: str = Field(..., description="Name of truncation nodes in the graph")
    edge_weight_attribute: str | None = Field(
        None, description="Name of the edge attribute to use as weights for the projections."
    )
    autocast: bool = Field(False, description="Whether to use autocasting for mixed precision training.")
