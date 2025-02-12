
import math

import numpy as np
import torch
from abc import ABC
import torch.nn as nn
from anemoi.models.layers.fourier import levels_expansion
import einops

class VerticalInformationEmbedder(nn.Module):
    def __init__(self, level_shuffle, method, fourier_dim, hidden_dim, encoded_dim, num_levels, data_indices) -> None:

        """Initialise.
        """
        super().__init__()
        self.level_shuffle = level_shuffle 
        self.data_indices = data_indices
        self.vertical_embeddings_method = method
        self.fourier_dim = fourier_dim
        self.hidden_dim = hidden_dim
        self.encoded_dim = encoded_dim
        self.num_levels = num_levels

        #if self.vertical_embeddings_method not in ['concat', 'addition']:
        #    raise ValueError(f"Unknown vertical embeddings method: {self.vertical_embeddings_method}")

        if self.vertical_embeddings_method == 'addition':
            assert encoded_dim == 1 , "For addition method, encoded_dim must be equal to 1"

        #! FOURIER AND ENCODED DIMENSIONS FOR PRESSURE LEVELS
        self.fourier_dim = fourier_dim # small otherwise CUDA OOM
        self.hidden_dim = hidden_dim
        self.encoded_dim = encoded_dim # small otherwise CUDA OOM
        self.mlp = self._define_mlp()


    def _get_levels_tensor(self)->torch.Tensor:
        level_list = []
        for var_str in self.data_indices.name_to_index:

            parts = var_str[0].split("_")
            # extract pressure level. If not pressure level, assume surface and assign 1000
            # TODO: make this dependent on new variable groups
            numeric_part = float(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 1000
            level_list.append(numeric_part)
        level_tensor = torch.tensor(level_list)
        if self.level_shuffle:
            level_tensor= level_tensor.reshape((self.num_variables,self.num_levels))[...,self.rand_idx].ravel()
        return level_tensor


    def _generate_shuffle_index(self):
        self.rand_idx = torch.randperm(self.num_levels)
        num_list = torch.Tensor([i for i in range(self.num_levels)])
        comb_list = torch.cat([self.rand_idx.view(self.num_levels, 1), num_list.view(self.num_levels, 1)], axis = 1)
        sorted_comb_list = comb_list[comb_list[:, 0].sort()[1]]
        self.rand_rev = [int(i) for i in sorted_comb_list[:, 1]]

    def _shuffle_input(self,x_reshaped):
        x_rand = x_reshaped[..., self.rand_idx]
        return x_rand

    def _define_mlp(self):
        # Construct MLP to transform vertical encodings
        act_func = nn.ReLU()
        mlp = nn.Sequential(nn.Linear(self.fourier_dim, self.hidden_dim), act_func)
        mlp.append(nn.Linear(self.hidden_dim, self.encoded_dim))
        mlp.append(act_func)
        return mlp


    def _encode_vertical_levels(self,level_tensor):
        #!1st ADDED Fourier transform
        vertical_encodings = levels_expansion(level_tensor, self.fourier_dim).to("cuda")

        #!2nd APPLY MLP
        mapped_vertical_features = self.mlp(vertical_encodings).ravel().to("cuda")
        return mapped_vertical_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        n_times = x.shape[1]
        ensemble_size = x.shape[2]
        num_grid_points = x.shape[3]        
        self.num_variables = int(x.shape[4]/self.num_levels)

        x_reshaped = torch.reshape(x, (batch_size, n_times, ensemble_size, num_grid_points, self.num_variables, self.num_levels))
        if self.level_shuffle:
            x_rand = self._shuffle_input(x_reshaped)
        else:
            x_rand = x_reshaped
    
        mapped_vertical_features= self._encode_vertical_levels(self._get_levels_tensor())
        if self.vertical_embeddings_method == 'concat':
            mapped_vertical_features = mapped_vertical_features.view(1, 1, 1, 1, mapped_vertical_features.shape[0])
            mapped_vertical_features= mapped_vertical_features.reshape(1,1,1,1,self.num_variables,self.num_levels,self.encoded_dim)
            mapped_vertical_features=mapped_vertical_features.expand(batch_size, n_times, ensemble_size, num_grid_points, self.num_variables, self.num_levels,self.encoded_dim)
            x_data_vertical_latent = torch.cat((x_rand.unsqueeze(-1), mapped_vertical_features), dim=-1)
            return einops.rearrange(
                    x_data_vertical_latent, "batch time ensemble grid vars levels concatdim -> (batch ensemble grid) (time vars levels concatdim)"
                )
        elif self.vertical_embeddings_method == 'addition':
            mapped_vertical_features = mapped_vertical_features.view(1, 1, 1, 1, self.num_variables, self.num_levels).expand(
                batch_size, n_times, 1, num_grid_points, self.num_variables, self.num_levels
            ) # ([4, 2, 1, 40320, 6, 13]
            x_data_vertical_latent = x_rand + mapped_vertical_features

            # pressure levels
            return einops.rearrange(
                    x_data_vertical_latent, "batch time ensemble grid vars levels -> (batch ensemble grid) (time vars levels)"
                )
        
        elif self.vertical_embeddings_method == 'attention':
            mapped_vertical_features = mapped_vertical_features.view(1, 1, 1, 1, self.num_variables, self.num_levels).expand(
                batch_size, n_times, 1, num_grid_points, self.num_variables, self.num_levels
            ) # ([4, 2, 1, 40320, 6, 13]
            return mapped_vertical_features
        else:
            return einops.rearrange(
                x_data_vertical_latent, "batch time ensemble grid vars levels -> (batch ensemble grid) (time vars levels)"
            )

