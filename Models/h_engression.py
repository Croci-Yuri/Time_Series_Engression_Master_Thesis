#################################################################################################################
                                            # Import libraries
#################################################################################################################

import torch
import torch.nn as nn
import os
import sys

# Set current and repository working directory
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from Modules.noise import Conditional_Noise_Encoder
from Modules.training_evaluation import *



#################################################################################################################
                                    # Heteroskedastic Engression
#################################################################################################################


class H_Engression(nn.Module):
    """
    Heteroskedastic Engression implements the pre-additive framework with input-dependent Gaussian noise.
    The noise scale σ(x) is learned through a conditional noise encoder and concatenated with the input
    before the multilayer perceptron.


    Args:
        input_dim (int): Number of input features.
        mlp_hidden_dim (int or list): Hidden layer size, or a list of sizes for each MLP layer.
        mlp_num_layers (int): Number of MLP layers when mlp_hidden_dim is an int (ignored if a list is provided).
        mlp_noise_dim (int): Dimensionality of the injected noise vector (must be > 0).
        mlp_noise_representation (str): Either 'scalar' or 'vector' for the conditional noise encoder.
        output_dim (int): Output dimensionality.
    """

    def __init__(self,
                 input_dim,
                 mlp_hidden_dim=100,
                 mlp_num_layers=2,
                 mlp_noise_dim=100,
                 mlp_noise_representation='scalar',
                 output_dim=1):

        super(H_Engression, self).__init__()

        # Basic noise arguments (checker)
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")
        if mlp_noise_dim <= 0:
            raise ValueError("mlp_noise_dim must be positive.")
        if mlp_noise_representation not in ("scalar", "vector"):
            raise ValueError("mlp_noise_representation must be 'scalar' or 'vector'.")

        # Layer widths (accept int or list/tuple)
        if isinstance(mlp_hidden_dim, int):
            mlp_hidden_dims = [mlp_hidden_dim]
        elif isinstance(mlp_hidden_dim, (list, tuple)) and len(mlp_hidden_dim) > 0:
            mlp_hidden_dims = list(mlp_hidden_dim)
        else:
            raise TypeError("mlp_hidden_dim must be an int or a nonempty list/tuple of ints.")
        mlp_num_layers = len(mlp_hidden_dims)

        self.mlp_noise_dim = mlp_noise_dim
        self.output_dim = output_dim

        ### Conditional noise encoder ###
        self.noise_generator = Conditional_Noise_Encoder(
            input_dim=input_dim,
            noise_dim=mlp_noise_dim,
            representation=mlp_noise_representation,
            sigma_min=1e-4, sigma_max=10.0 # safety bounds
        )

        ### MLP layers architecture ###
        self.mlp_layers = nn.ModuleList()
        for layer_idx in range(mlp_num_layers):
            if layer_idx == 0:
                in_dim = input_dim + self.mlp_noise_dim
            else:
                in_dim = mlp_hidden_dims[layer_idx - 1]

            out_dim = mlp_hidden_dims[layer_idx]
            layers = [nn.Linear(in_dim, out_dim), nn.ReLU()]
            self.mlp_layers.append(nn.Sequential(*layers))

        ### Output layer ###
        self.output_layer = nn.Linear(mlp_hidden_dims[-1], output_dim)

    ### Forward pass ###
    def forward(self, x):
        batch_size = x.size(0)
        x_in = x

        for i, layer in enumerate(self.mlp_layers):

            # Noise injection first layer + concatenation with input space
            if i == 0:
                noise = self.noise_generator(x)  
                x_in = torch.cat([x_in, noise], dim=-1)

            # Simple forward pass to next layer
            x_in = layer(x_in)

        return self.output_layer(x_in)

#################################################################################################################