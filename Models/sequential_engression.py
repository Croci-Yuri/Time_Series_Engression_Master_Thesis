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


from Modules.training_evaluation import *


#################################################################################################################
                            # Sequential Engression (GRU encoder + Engression decoder)
#################################################################################################################


class Sequential_Engression(nn.Module):
    """
    Sequential Engression implements the pre-additive framework with temporal encoding.
    A GRU encoder processes the input sequence, after which Gaussian noise is concatenated
    with the pooled representation before the multilayer perceptron.


    Args:
        input_dim (int): Number of input features per time step.
        gru_hidden_size (int or list): Hidden units per GRU layer (int converted to single-layer list).
        pooling (str): 'last' or 'static_softmax'.
        sequence_length (int): Length of the input sequence.

        mlp_hidden_dim (int or list): Hidden layer size, or a list of sizes for each MLP layer.
        mlp_sigma (float): Standard deviation of Gaussian noise (must be > 0).
        mlp_noise_dim (int): Dimensionality of the injected noise vector (must be > 0).

        output_dim (int): Output dimensionality.
    """

    def __init__(self,
                 input_dim,
                 gru_hidden_size,
                 pooling='last',
                 sequence_length=5,
                 mlp_hidden_dim=100,
                 mlp_sigma=1,
                 mlp_noise_dim=100,
                 output_dim=1):

        super(Sequential_Engression, self).__init__()

        # Basic arguments (checker)
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")
        if pooling not in {"last", "static_softmax"}:
            raise ValueError("pooling must be 'last' or 'static_softmax'.")
        if mlp_sigma <= 0 or mlp_noise_dim <= 0:
            raise ValueError("mlp_sigma and mlp_noise_dim must be positive.")

        self.pooling = pooling
        self.sequence_length = sequence_length
        self.mlp_sigma = mlp_sigma
        self.mlp_noise_dim = mlp_noise_dim
        self.output_dim = output_dim

        # GRU widths (accept int or list/tuple)
        if isinstance(gru_hidden_size, int):
            gru_sizes = [gru_hidden_size]
        elif isinstance(gru_hidden_size, (list, tuple)) and len(gru_hidden_size) > 0:
            gru_sizes = list(gru_hidden_size)
        else:
            raise TypeError("gru_hidden_size must be an int or a nonempty list/tuple of ints.")

        ### Temporal Encoder (GRU) layers architecture ###
        self.gru_layers = nn.ModuleList()
        in_dim = input_dim
        for h in gru_sizes:
            self.gru_layers.append(nn.GRU(in_dim, h, batch_first=True))
            in_dim = h
        self.gru_output_size = gru_sizes[-1]

        # Static softmax pooling parameters initialization
        if self.pooling == "static_softmax":
            self.lag_scores = nn.Parameter(torch.zeros(self.sequence_length))

        # MLP widths (accept int or list/tuple)
        if isinstance(mlp_hidden_dim, int):
            mlp_hidden_dims = [mlp_hidden_dim]
        elif isinstance(mlp_hidden_dim, (list, tuple)) and len(mlp_hidden_dim) > 0:
            mlp_hidden_dims = list(mlp_hidden_dim)
        else:
            raise TypeError("mlp_hidden_dim must be an int or a nonempty list/tuple of ints.")

        ### MLP Decoder layers architecture ###
        self.mlp_layers = nn.ModuleList()
        for layer_idx, out_dim in enumerate(mlp_hidden_dims):
            in_dim = (self.gru_output_size + self.mlp_noise_dim) if layer_idx == 0 else mlp_hidden_dims[layer_idx - 1]
            self.mlp_layers.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU()))

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

        ### MLP decoding with noise concatenation ###
        z = h_pooled
        for i, layer in enumerate(self.mlp_layers):
            if i == 0:
                new_noise = torch.randn(batch_size, self.mlp_noise_dim, device=z.device, dtype=z.dtype) * self.mlp_sigma
                z = torch.cat([z, new_noise], dim=-1)
            z = layer(z)

        return self.output_layer(z)

#################################################################################################################