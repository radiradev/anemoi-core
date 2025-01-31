
import math

import numpy as np
import torch
import torch.nn as nn
from anemoi.models.layers.fourier import levels_expansion


class VerticalInformationEmbedder(nn.Module):

    def __init__(self, config:dict,data_indices) -> None:
        """Initialise.
        """
        super().__init__()
        self.shuffling = config.level_shuffle 
        self.method = config.method
        self.data_indices = data_indices
        N_ATMOS_VARIABLES = 6

        #! FOURIER AND ENCODED DIMENSIONS FOR PRESSURE LEVELS
        self.fourier_dim = config.fourier_dim # small otherwise CUDA OOM
        self.hidden_dim = configs.hidden_dim
        self.encoded_dim =config.encoded_dim # small otherwise CUDA OOM

        self.mlp = self._define_mlp()


    def _get_levels_tensor(self)->torch.Tensor:
        level_list = []
        for var_str in self.data_indices:
            parts = var_str[0].split("_")
            # extract pressure level. If not pressure level, assume surface and assign 1000
            # TODO: make this dependent on new variable groups
            numeric_part = float(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 1000
            level_list.append(numeric_part)
        level_tensor = torch.tensor(level_list)
        if self.level_shuffle:
            level_tensor= level_tensor.reshape((N_ATMOS_VARIABLES,self.num_levels))[...,self.rand_idx].ravel()
        return level_tensor

    def _shuffle_input(self):
        self.rand_idx = torch.randperm(self.num_levels)
        num_list = torch.Tensor([i for i in range(self.num_levels)])
        comb_list = torch.cat([rand_idx.view(self.num_levels, 1), num_list.view(self.num_levels, 1)], axis = 1)
        sorted_comb_list = comb_list[comb_list[:, 0].sort()[1]]
        self.rand_rev = [int(i) for i in sorted_comb_list[:, 1]]
        x_rand = x_reshaped[..., rand_idx]

    def _define_mlp(self):
        # Construct MLP to transform vertical encodings
        act_func = nn.ReLU()
        mlp = nn.Sequential(nn.Linear(self.fourier_dim, self.hidden_dim), act_func)
        mlp.append(nn.Linear(self.hidden_dim, self.encoded_dim))
        mlp.append(act_func)
        return mlp


    def _encode_vertical_levels(self,level_tensor):
        #!1st ADDED Fourier transform
        vertical_encodings = levels_expansion(level_tensor, self.fourier_dim)

        #!2nd APPLY MLP
        mapped_vertical_features = self.mlp(vertical_encodings).ravel().to("cuda")
        return mapped_vertical_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        batch_size = x.shape[0]
        n_times = x.shape[1]
        ensemble_size = x.shape[2]
        num_grid_points = x.shape[3]        
        num_variables = int(x.shape[4]/self.num_levels)

        x_reshaped = torch.reshape(x, (batch_size, n_times, ensemble_size, num_grid_points, num_variables, self.num_levels))
        if self.level_shuffle:
            x_rand = self._shuffle_input(x_reshaped)
        else:
            x_rand = x_reshaped
    
        mapped_vertical_features= self._encode_vertical_levels(self._get_levels_tensor())

        if self.vertical_embeddings_method == 'concat':
            ## TODO: Can you check this please Ana?
            mapped_vertical_features = mapped_vertical_features.view(1, 1, 1, 1, num_variables, self.num_levels).expand(
                batch_size, n_times, 1, num_grid_points, num_variables, self.num_levels, -1
            ) # ([4, 2, 1, 40320, 6, 13, 99*self.encoded_dim]
            x_data_vertical_latent = torch.cat((x, mapped_vertical_features), dim=-1)
        elif self.vertical_embeddings_method == 'addition':
            mapped_vertical_features = mapped_vertical_features.view(1, 1, 1, 1, num_variables, self.num_levels).expand(
                batch_size, n_times, 1, num_grid_points, num_variables, self.num_levels
            ) # ([4, 2, 1, 40320, 6, 13]
            x_data_vertical_latent = x_rand + mapped_vertical_features
        else:
            raise ValueError(f"Unknown vertical embeddings method: {self.vertical_embeddings_method}")

        return x_data_vertical_latent