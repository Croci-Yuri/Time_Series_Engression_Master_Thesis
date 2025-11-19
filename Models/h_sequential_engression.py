#################################################################################################################
                                            # Import libraries
#################################################################################################################

import torch
import torch.nn as nn
import sys
import os

# Set current and repository working directory
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from Modules.noise import Conditional_Noise_Encoder
from Modules.training_evaluation import *


#################################################################################################################
                                # Heteroskedastic SML-Engression with learned noise injection
#################################################################################################################

class H_Sequential_Engression(nn.Module):
    """
    Heteroskedastic Sequential Engression implements the pre-additive framework with temporal
    encoding and input-dependent Gaussian noise.
    A GRU encoder produces a pooled representation, and a conditional noise encoder learns σ(h)
    to generate noise that is concatenated before the multilayer perceptron.

    Args:
        input_dim (int): Number of input features per time step.
        gru_hidden_size (int or list): Hidden units per GRU layer, or list of sizes for each GRU layer.
        pooling (str): 'last' or 'static_softmax'.
        sequence_length (int): Length of the input sequence.

        mlp_hidden_dim (int or list): Hidden layer size, or a list of sizes for each MLP layer.
        mlp_noise_dim (int): Dimensionality of the injected noise vector (must be > 0).
        mlp_noise_representation (str): 'scalar' or 'vector' representation for the Conditional Noise Encoder.

        output_dim (int): Output dimensionality.
    """

    def __init__(self,
                 input_dim,
                 gru_hidden_size,
                 pooling='last',
                 sequence_length=5,
                 mlp_hidden_dim=100,
                 mlp_noise_dim=100,
                 mlp_noise_representation='scalar',
                 output_dim=1):

        super(H_Sequential_Engression, self).__init__()

        # Basic noise arguments (checker)
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")
        if pooling not in {"last", "static_softmax"}:
            raise ValueError("pooling must be 'last' or 'static_softmax'.")
        if mlp_noise_dim <= 0:
            raise ValueError("mlp_noise_dim must be positive.")
        if mlp_noise_representation not in ("scalar", "vector"):
            raise ValueError("mlp_noise_representation must be 'scalar' or 'vector'.")

        self.pooling = pooling
        self.sequence_length = sequence_length
        self.mlp_noise_dim = mlp_noise_dim
        self.output_dim = output_dim

        # GRU widths (accept int or list/tuple)
        if isinstance(gru_hidden_size, int):
            gru_sizes = [gru_hidden_size]
        elif isinstance(gru_hidden_size, (list, tuple)) and len(gru_hidden_size) > 0:
            gru_sizes = list(gru_hidden_size)
        else:
            raise TypeError("gru_hidden_size must be an int or a nonempty list/tuple of ints.")

        ### Temporal Encoder (GRU) layers architecture
        self.gru_layers = nn.ModuleList()
        in_dim = input_dim
        for h in gru_sizes:
            self.gru_layers.append(nn.GRU(in_dim, h, batch_first=True))
            in_dim = h
        self.gru_output_size = gru_sizes[-1]

        # Static softmax pooling parameters initialization
        if self.pooling == "static_softmax":
            self.lag_scores = nn.Parameter(torch.zeros(self.sequence_length))

        ### Conditional noise encoder on pooled representation ###
        self.noise_encoder = Conditional_Noise_Encoder(
            input_dim=self.gru_output_size,
            noise_dim=mlp_noise_dim,
            representation=mlp_noise_representation,
            sigma_min=1e-4, sigma_max=10.0  # safety bounds
        )

        # MLP widths (accept int or list/tuple)
        if isinstance(mlp_hidden_dim, int):
            mlp_hidden_dims = [mlp_hidden_dim]
        elif isinstance(mlp_hidden_dim, (list, tuple)) and len(mlp_hidden_dim) > 0:
            mlp_hidden_dims = list(mlp_hidden_dim)
        else:
            raise TypeError("mlp_hidden_dim must be an int or a nonempty list/tuple of ints.")
        mlp_num_layers = len(mlp_hidden_dims)

        ### MLP Decoder layers architecture ###
        self.mlp_layers = nn.ModuleList()
        for layer_idx in range(mlp_num_layers):
            if layer_idx == 0:
                in_dim = self.gru_output_size + self.mlp_noise_dim
            else:
                in_dim = mlp_hidden_dims[layer_idx - 1]

            out_dim = mlp_hidden_dims[layer_idx]
            layers = [nn.Linear(in_dim, out_dim), nn.ReLU()]
            self.mlp_layers.append(nn.Sequential(*layers))

        ### Output layer ###
        self.output_layer = nn.Linear(mlp_hidden_dims[-1], output_dim)

    ### Forward pass ###
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        if seq_len != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {seq_len}.")

        # Temporal encoding
        for rnn in self.gru_layers:
            x, h_t = rnn(x)
        seq_out = x

        ### Pooling layer ###
        if self.pooling == "last":
            h_pooled = h_t[-1]
        else:  # static_softmax
            weights = torch.softmax(self.lag_scores.to(x.device), dim=0).view(1, -1, 1)
            h_pooled = (seq_out * weights).sum(dim=1)

        ### Learned noise concatenation at first MLP layer ###
        z = h_pooled
        for i, layer in enumerate(self.mlp_layers):
            if i == 0:
                noise = self.noise_encoder(h_pooled)
                z = torch.cat([z, noise], dim=-1)

            # Simple forward pass to next layer
            z = layer(z)

        return self.output_layer(z)

#################################################################################################################