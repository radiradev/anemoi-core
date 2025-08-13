# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging

import torch

from anemoi.training.losses.weightedloss import BaseWeightedLoss

LOGGER = logging.getLogger(__name__)


class WeightedAreaRelatetSortedIntensityLoss(BaseWeightedLoss):
    """Node-weighted fuzzy MSE loss."""

    name = "arsil"

    def __init__(
        self,
        node_weights: torch.Tensor,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Node- and feature weighted MSE Loss.

        Parameters
        ----------
        node_weights : torch.Tensor of shape (N, )
            Weight of each node in the loss function
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False

        """
        super().__init__(
            node_weights=node_weights,
            ignore_nans=ignore_nans,
        )
        # register some kwargs as properties
        for key in ["num_neighbors", "depth"]:
            if key in kwargs: setattr(self, key, kwargs[key]) 

        # aggregate node_weights to have the correct shape
        # self.node_weights = self._aggregate_node_weights(
        #     self.node_weights, 
        #     num_neighbors=self.num_neighbors, 
        #     depth=self.depth,
        # ) 
        #TODO: remove original_node_weights and determine node_weights
        # of correct shape only once, of no change of depth during run time necessary
        self.register_buffer("original_node_weights", self.node_weights, persistent=True)

    def _aggregate(self,
                  x: torch.Tensor, 
                  num_neighbors:int = 4,
                  depth: int = 1,
                  sort: bool = False,
                  top_level: bool = True,
    ) -> torch.Tensor:
        """aggregates and sorts features in a local area 

        Parameters
        ----------
        x : torch.Tensor    
            Either pred or target shape (bs, ensemble, lat*lon, n_outputs)
            or (ensemble, lat*lon, n_outputs)
        num_neighbors : int 
            number of neighbors to be aggregated in one super-location
        depth : int 
            number of aggreatation steps -> depth of recursion
        sort : bool 
            debugging setting -> sort=False -> aggreatation has no effect
        top_level : bool
            Determines whether recursive routine is on outer most level

        Returns
        -------
        x: torch.Tensor 
            aggregated/reshaped pred or target
            shape : (bs, ensemble, lat*lon/num_neighbors, num_neighbors, n_outputs)
            or    : (ensemble, lat*lon/num_neighbors, num_neighbors, n_outputs)
        """
        # determine if batch dimension present
        no_batch = True if top_level and len(x.shape)==3 else False
        # if no batch dimension create trivial 
        if no_batch: x = x.unsqueeze(0)

        # go deeper in recursion?
        if depth > 1:
            x = self._aggregate(x, num_neighbors, depth=depth - 1, top_level=False)
            depth -= 1

        # aggretate via reshaping (4 rows always grouped spatially by construction)
        N = x.shape[2]
        latent = x.shape[3] if len(x.shape) > 4 else 1
        N_target = N // num_neighbors
        x = x.reshape(x.shape[0], x.shape[1], N_target, num_neighbors * latent, x.shape[-1])
        if sort: x, _ = torch.sort(x, dim=-2)

        # remove batch dimension if trivial
        if no_batch: x = x.squeeze(0)

        return x 

    def _aggregate_node_weights(self,
                                x: torch.Tensor,
                                num_neighbors: int = 4,
                                depth: int = 1,
    ) -> torch.Tensor:
        """applies spatial aggregation to node_weights

        Parameters
        ----------
        x : torch.Tensor    
            node_weights shape (lat*lon) or (lat*lon/N, N)
        num_neighbors : int 
            number of neighbors to be aggregated in one super-location
        depth : int 
            number of aggreatation steps -> depth of recursion
        sort : bool 
            debugging setting -> sort=False -> aggreatation has no effect

        Returns
        -------
        x: torch.Tensor 
            aggregated/reshaped pred or target
            shape : (lat*lon/Nnneigh, num_neighbors)

        """

        if depth > 1:
            x = self._aggregate_node_weights(x, num_neighbors, depth=depth - 1)
            depth -= 1
        N = x.shape[0]
        latent = x.shape[1] if len(x.shape) > 1 else 1
        N_target = N // num_neighbors
        x = x.reshape(N_target, num_neighbors * latent)
        return x

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        scalar_indices: tuple[int, ...] | None = None,
        without_scalars: list[str] | list[int] | None = None,
        depth: int = 1,
    ) -> torch.Tensor:
        """Calculates the lat-weighted MSE loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True
        scalar_indices: tuple[int,...], optional
            Indices to subset the calculated scalar with, by default None
        without_scalars: list[str] | list[int] | None, optional
            list of scalars to exclude from scaling. Can be list of names or dimensions to exclude.
            By default None
        depth: int, optional
            number of recursive coarsening steps in area related sorting
        Returns
        -------
        torch.Tensor
            Weighted ARSIL loss
        """

        # do aggretation
        pred = self._aggregate(pred, num_neighbors=self.num_neighbors, depth=self.depth, sort=True) 
        target = self._aggregate(target, num_neighbors=self.num_neighbors, depth=self.depth, sort=True)

        #TODO: original_node_weights should become unneccary if no regular change
        # of depth is needed, otherwise just determine reshpaed node_weights once
        # Further the sorting of node_weights is not uniquely defined, since pred 
        # and target are sorted differently -> solution: define coarse node_weights
        # appropriate for target resolution where loss is defined deterministically
        self.node_weights = self._aggregate_node_weights(
            self.original_node_weights, 
            num_neighbors=self.num_neighbors, 
            depth=depth,
            sort=False,
        ) 

        out = torch.square(pred - target)
        out = self.scale(out, scalar_indices, without_scalars=without_scalars)
        return self.scale_by_node_weights(out, squash)
