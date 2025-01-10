# (C) Copyright 2024-2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

from .edge_schemas import EdgeAttributeSchema
from .edge_schemas import EdgeBuilderSchemas
from .node_schemas import NodeAttributeSchemas
from .node_schemas import NodeBuilderSchemas

LOGGER = logging.getLogger(__name__)


class NodeSchema(BaseModel):
    node_builder: NodeBuilderSchemas | Any
    "Node builder schema."
    attributes: dict[str, NodeAttributeSchemas] | None = None
    "Dictionary of attributes with names as keys and anemoi.graph.nodes.attributes objects as values."

    @field_validator("node_builder")
    @classmethod
    def validate_nodebuilder(cls, node_builder: NodeBuilderSchemas | Any) -> NodeBuilderSchemas | Any:
        if not isinstance(node_builder, NodeBuilderSchemas):
            LOGGER.warning("%s not defined in anemoi.", node_builder)
        return node_builder


class EdgeSchema(BaseModel):
    source_name: str = Field(examples=["data", "hidden"])
    "Source of the edges."
    target_name: str = Field(examples=["data", "hidden"])
    "Target of the edges."
    edge_builders: list[EdgeBuilderSchemas | Any]
    "Edge builder schema."
    attributes: dict[str, EdgeAttributeSchema]
    "Dictionary of attributes with names as keys and anemoi.graph.edges.attributes objects as values."

    @field_validator("edge_builders")
    @classmethod
    def validate_edgebuilders(cls, edge_builders: list) -> list:
        for edge_builder in edge_builders:
            if not isinstance(edge_builder, EdgeBuilderSchemas):
                LOGGER.warning("%s not defined in anemoi.", edge_builder)
        return edge_builders


class BaseGraphSchema(BaseModel):
    nodes: dict[str, NodeSchema]
    "Nodes schema for all types of nodes (ex. data, hidden)."
    edges: list[EdgeSchema]
    "List of edges schema."
    overwrite: bool = Field(default=True)
    "whether to overwrite existing graph file. Default to True."
    data: str = Field(default="data")
    "Key name for the data nodes. Default to 'data'."
    hidden: str = Field(default="hidden")
    "Key name for the hidden nodes. Default to 'hidden'."
    # TODO(Helen): Needs to be adjusted for more complex graph setups
