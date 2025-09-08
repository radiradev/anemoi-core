# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import uuid
from typing import Optional

import einops
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.projection import NodeEmbedder
from anemoi.models.layers.projection import NodeProjector
from anemoi.models.preprocessing.normalisers import build_normaliser
from anemoi.utils.config import DotDict

from .base import AnemoiModel

LOGGER = logging.getLogger(__name__)


NODE_COORDS_NDIMS = 4  # cos_lat, sin_lat, cos_lon, sin_lon
EDGE_ATTR_NDIM = 3  # edge_length, edge_dir0, edge_dir1


def processor_factory(name_to_index, statistics, processors, **kwargs) -> list[list]:

    return [
        [name, instantiate(cfg, name_to_index=name_to_index["variables"], statistics=statistics["variables"])]
        for name, cfg in processors.items()
    ]


def extract_sources(config, reversed: bool = False) -> tuple[dict, dict[str, list[str]]]:
    mapper_config, sources, num_channels = {}, {}, {}
    for i, component in enumerate(config):
        name = component.pop("name", f"{i+1}")
        for source in component.pop("sources"):
            if reversed:
                sources[name] = source
            else:
                sources[source] = name
        mapper_config[name] = component["mapper"]
        num_channels[name] = component["num_channels"]
    return mapper_config, sources, num_channels


def merge_nodes(graph: HeteroData, merged_name: str, nodes_names: list[str]) -> tuple[HeteroData, dict]:
    if len(nodes_names) == 1:
        graph = graph.rename(nodes_names[0], merged_name)
        slices = {nodes_names[0]: slice(0, graph[nodes_names[0]].num_nodes)}
        return graph, slices

    num_nodes = {name: graph[name].num_nodes for name in nodes_names}
    graph[merged_name].x = torch.cat([graph[nodes].x for nodes in nodes_names])
    # TODO: Merge edge_index of all subgraphs
    slices, count = {}, 0
    for nodes in nodes_names:
        num_nodes = graph[nodes].num_nodes
        slices[nodes] = slice(count, count + num_nodes)
        count += num_nodes
    return graph, slices


def merge_graph_sources(graph: HeteroData, sources: dict[str, str]) -> HeteroData:
    if sources is None or len(sources) == 0:
        return graph, None

    graph = graph.clone()
    new = {}
    for k, v in sources.items():
        new[v] = (new[v] + [k]) if v in new else [k]

    slices = {}
    for new_node_names, old_node_names in new.items():
        graph, nodes_slices = merge_nodes(graph, new_node_names, old_node_names)
        slices[new_node_names] = nodes_slices

    return graph, slices


def num_channels(name_to_index, **kwargs):
    import warnings

    warnings.warn("assuming only one offset per tensor")
    num_channels = len(name_to_index)
    if "add_channels" in kwargs:
        num_channels += kwargs["add_channels"]
    return num_channels


class AnemoiMultiModel(AnemoiModel):
    """Message passing graph neural network."""

    def __init__(self, *, sample_static_info: "Structure", model_config: DotDict, metadata) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        model_config : DotDict
            Model configuration
        graph_data : HeteroData
            Graph definition
        """
        print(f"✅ model : {self.__class__.__name__}")
        super().__init__()
        self.id = str(uuid.uuid4())
        self.metadata = metadata

        self.supporting_arrays = {}
        model_config = DotDict(model_config)

        self.sample_static_info = sample_static_info

        # Instantiate processors
        self.normaliser = self.sample_static_info.create_function(build_normaliser)

        # # apply is not supported anymore
        # preprocessors = self.sample_static_info.apply(processor_factory)
        # # Assign the processor list pre- and post-processors
        # self.input_pre_processors = Processors(preprocessors["input"].processor_factory)
        # self.target_pre_processors = Processors(preprocessors["target"].processor_factory)
        # self.target_post_processors = Processors(preprocessors["target"].processor_factory, inverse=True)
        # # TODO: Implemente structure.processor_factory (not only at LeafStructure)

        self.num_channels = model_config.model.num_channels

        self.latent_residual_connection = True
        self.merge_latents_method = model_config.model.model.merge_latents

        if model_config.model.get("residual_connections"):
            warnings.warn("Residual connections not supported")
        # def _define_residual_connection_indices(
        #     input,
        #     target,
        #     residual_connections: list[str],
        # ):
        #     residual_connection_indices = {}
        #     for source in residual_connections:
        #         input_vars = input_name_to_index[source]
        #         target_vars = target_name_to_index[source]
        #         common_vars = set(target_vars).intersection(input_vars)
        #         target_idx = [input_vars[v] for v in common_vars]
        #         input_idx = [target_vars[v] for v in common_vars]
        #         residual_connection_indices[source] = target_idx, input_idx
        #     return residual_connection_indices

        # self.residual_connection_indices = _define_residual_connection_indices(
        #     sample_static_info["input"],
        #     sample_static_info["target"],
        #     residual_connections=model_config.model.get("residual_connections", []),
        # )

        # NODE_COORDS_NDIMS = 4  # cos_lat, sin_lat, cos_lon, sin_lon
        # should be in the input ?
        self.num_input_channels = sample_static_info.input.box_to_any(num_channels)
        self.num_target_channels = sample_static_info.target.box_to_any(num_channels)

        self.hidden_name: str = model_config.model.model.hidden_name
        encoders, self.encoder_sources, num_encoded_channels = extract_sources(model_config.model.model.encoders)
        decoders, self.decoder_sources, num_decoded_channels = extract_sources(
            model_config.model.model.decoders, reversed=True
        )

        # Embedding layers
        self.node_embedder = NodeEmbedder(
            model_config.model.model.emb_data,
            num_input_channels=self.num_input_channels,
            num_output_channels=num_encoded_channels,
            sources=self.encoder_sources,
        )
        self.node_projector = NodeProjector(
            model_config.model.model.emb_data,
            num_input_channels=num_decoded_channels,
            num_output_channels=self.num_target_channels,
            sources=self.decoder_sources,
        )

        # Encoders: ??? -> hidden
        self.encoders = nn.ModuleDict({})
        for enc_name, encoder_config in encoders.items():
            self.encoders[enc_name] = instantiate(
                encoder_config,
                _recursive_=False,
                in_channels_src=num_encoded_channels[enc_name],
                in_channels_dst=NODE_COORDS_NDIMS,
                hidden_dim=num_encoded_channels[enc_name],
                edge_dim=EDGE_ATTR_NDIM,
            )

        # Processor hidden -> hidden
        self.processor = instantiate(
            model_config.model.model.processor,
            _recursive_=False,
            num_channels=self.num_channels,
            edge_dim=EDGE_ATTR_NDIM,
        )

        # Decoders: hidden -> ???
        self.decoders = nn.ModuleDict({})
        for dec_name, decoder_config in decoders.items():
            self.decoders[dec_name] = instantiate(
                decoder_config,
                _recursive_=False,
                in_channels_src=self.num_channels,
                in_channels_dst=NODE_COORDS_NDIMS,
                hidden_dim=num_decoded_channels[dec_name],
                out_channels_dst=num_decoded_channels[dec_name],
                edge_dim=EDGE_ATTR_NDIM,
            )

        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        # TODO: Bring bounding back
        self.boundings = nn.ModuleList([])
        # self.boundings = nn.ModuleList(
        #    [
        #        instantiate(
        #            cfg,
        #            name_to_index=sample_static_info.target.name_to_index,
        #            statistics=sample_static_info.input.statistics,
        #            name_to_index_stats=sample_static_info.input.name_to_index,
        #        )
        #        for cfg in getattr(model_config.model, "bounding", [])
        #    ]
        # )

    def _assemble_dict(
        self, x: dict[str, torch.Tensor], graph: HeteroData, batch_size: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # IN: x.values() shape: (batch_size, multi_step, ens_dim, grid_size, num_vars)
        # OUT: x_data.values() shape: ( (batch_size * ens_dim * grid_size), (multi_step * num_vars) )

        x_data, x_data_skip = {}, {}
        for name, x_source_raw in x.items():
            x_data[name] = self._assemble_tensor(name, x_source_raw, graph, batch_size=batch_size)
            x_data[name] = torch.cat(
                (
                    einops.rearrange(
                        x_source_raw, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"
                    ),
                    self.get_node_coords(graph, name),
                    # TODO: add node trainable parameters
                ),
                dim=-1,  # feature dimension
            )

            x_data_skip[name] = x_source_raw[:, -1, :, :, :]

        return x_data, x_data_skip

    def _assemble_tensor(
        self, name: str, x: torch.Tensor, graph: HeteroData, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_dst_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                self.get_node_coords(graph, name),
                # TODO: add node trainable parameters
            ),
            dim=-1,  # feature dimension
        )
        return x_dst_data_latent

    def _run_mapper(
        self,
        mapper: nn.Module,
        data: tuple[Tensor],
        sub_graph,
        batch_size: int,
        shard_shapes: tuple[tuple[int, int], tuple[int, int]],
        model_comm_group: Optional[ProcessGroup] = None,
        use_reentrant: bool = False,
    ) -> Tensor:
        """Run mapper with activation checkpoint.

        Parameters
        ----------
        mapper : nn.Module
            Which processor to use
        data : tuple[Tensor]
            tuple of data to pass in
        sub_graph
            Sub graph to use for the mapper
        batch_size: int,
            Batch size
        shard_shapes : tuple[tuple[int, int], tuple[int, int]]
            Shard shapes for the data
        model_comm_group : ProcessGroup
            model communication group, specifies which GPUs work together
            in one model instance
        use_reentrant : bool, optional
            Use reentrant, by default False

        Returns
        -------
        Tensor
            Mapped data
        """
        return checkpoint(
            mapper,
            data,
            sub_graph,
            batch_size=batch_size,
            shard_shapes=shard_shapes,
            model_comm_group=model_comm_group,
            use_reentrant=use_reentrant,
        )

    def encode(
        self,
        x: dict[str, torch.Tensor],
        graph: HeteroData,
        shard_shapes_hidden: tuple[list],
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        x_data_raw, x_hidden_raw = x
        x_data_latents = self.node_embedder(x_data_raw)

        # TODO: Merge subgraph
        graph, _ = merge_graph_sources(graph, self.encoder_sources)

        x_hidden_latents = {}
        for encoder_name, x_data_latent in x_data_latents.items():
            shard_shapes_input_data = get_shard_shapes(x_data_latent, 0, model_comm_group)

            _, x_hidden_latents[encoder_name] = self._run_mapper(
                self.encoders[encoder_name],
                (x_data_latent, x_hidden_raw),
                sub_graph=graph[(encoder_name, "to", self.hidden_name)].to(x_data_latent.device),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_input_data, shard_shapes_hidden),
                model_comm_group=model_comm_group,
            )

        x_hidden_latent = self.merge_latents(x_hidden_latents)

        return x_data_latents, x_hidden_latent

    def merge_latents(self, latents: dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO: implement different strategies: sum, average, concat, learnable, ...
        return latents[list(latents.keys())[0]]

    def decode(
        self,
        x: tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]],
        graph: HeteroData,
        shard_shapes_hidden: tuple[list],
        batch_size: int,
        ensemble_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ):
        x_hidden_latent, x_raw_target = x

        sources = {v: k for k, v in self.decoder_sources.items()}
        graph, node_slices = merge_graph_sources(graph, sources)

        x_out = {}
        for dec_name in self.decoders.keys():
            if dec_name in x_raw_target:
                x_target_latent = self._assemble_tensor(dec_name, x_raw_target[dec_name], graph, batch_size=batch_size)
            else:
                x_target_latent = self.get_node_coords(graph, dec_name)

            shard_shapes_target_data = get_shard_shapes(x_target_latent, 0, model_comm_group)
            # This may be passed when name in x_target_data

            x_out[dec_name] = self._run_mapper(
                self.decoders[dec_name],
                (x_hidden_latent, x_target_latent),
                sub_graph=graph[(self.hidden_name, "to", dec_name)].to(x_hidden_latent.device),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hidden, shard_shapes_target_data),
                model_comm_group=model_comm_group,
            )

        x_out = self.node_projector(x_out, node_slices)

        return x_out

    def residual_connection(
        self, y: dict[str, torch.Tensor], x_skips: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        for source, (target_idx, input_idx) in self.residual_connection_indices.items():
            # residual connection (just for the prognostic variables)
            y[source][..., target_idx] += x_skips[source][..., input_idx]

        return y

    def bound(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        y = {}
        for tensor_name, pred in x.items():
            for bounding in self.boundings:
                # bounding performed in the order specified in the config file
                pred = bounding(pred)

            y[tensor_name] = pred

        return y

    def get_node_coords(self, graph: HeteroData, name: str) -> torch.Tensor:
        assert name in graph.node_types, f"{name} do not exist. Valid graph nodes: {graph.node_types}."
        return torch.cat([torch.sin(graph[name].x), torch.cos(graph[name].x)], dim=-1)

    def forward(
        self, x: dict[str, Tensor], graph: HeteroData, *, model_comm_group: Optional[ProcessGroup] = None, **kwargs
    ) -> dict[str, Tensor]:
        # at this point, the input (x) has already been normalised
        # if this is not wanted, don't normalise it in the task
        x = self.sample_static_info.merge_content(x)

        print(x.to_str("Input Batch"))

        batch_size = x[list(x.keys())[0]]["data"].shape[0]
        ensemble_size = 1

        x_hidden = self.get_node_coords(graph, self.hidden_name)
        shard_shapes_hidden = get_shard_shapes(x_hidden, 0, model_comm_group)

        x_data_latents, x_data_skip = self._assemble_dict(x, graph, batch_size=batch_size)

        x_data_latents, x_hidden_latent = self.encode(
            (x_data_latents, x_hidden),
            graph,
            batch_size=batch_size,
            shard_shapes_hidden=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        x_latent_proc = self.processor(
            x_hidden_latent,
            graph[(self.hidden_name, "to", self.hidden_name)].to(x_hidden_latent.device),
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        if self.latent_residual_connection:
            x_latent_proc = x_latent_proc + x_hidden_latent

        x_out = self.decode(
            (x_latent_proc, x_data_latents),
            graph,
            shard_shapes_hidden=shard_shapes_hidden,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            model_comm_group=model_comm_group,
        )

        x_out = self.residual_connection(x_out, x_data_skip)
        x_out = self.bound(x_out)

        return x_out
