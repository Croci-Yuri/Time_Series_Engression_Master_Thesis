#################################################################################################################
                                        # Import libraries and modules
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


from Modules.Early_Stopping import *
from Modules.training_evaluation_deterministic import *



#################################################################################################################
                                        # Sequential MLP
#################################################################################################################

class Sequential_MLP(nn.Module):
    """
    Deterministic GRU encoder with either 'last' or 'static_softmax' pooling,
    followed by an MLP decoder.

    Args:
        input_size (int): Number of input features.
        gru_hidden_size (int or list): Hidden units per GRU layer (or list of sizes per layer).
        output_size (int): Dimensionality of the output.

        pooling (str): 'last' or 'static_softmax'.
        sequence_length (int): Length of input sequences (required for static_softmax).

        mlp_hidden_size (int or list): Hidden units in the MLP or list of layer sizes.
    """

    def __init__(self,
                 input_size,
                 gru_hidden_size,
                 output_size,
                 pooling='last',
                 sequence_length=5,
                 mlp_hidden_size=64):

        super().__init__()

        # Pooling (checker)
        if pooling not in {"last", "static_softmax"}:
            raise ValueError("pooling must be 'last' or 'static_softmax'.")
        self.pooling = pooling
        self.sequence_length = sequence_length

        # GRU widths (accept int or list/tuple)
        if isinstance(gru_hidden_size, int):
            rnn_sizes = [gru_hidden_size]
        elif isinstance(gru_hidden_size, (list, tuple)) and len(gru_hidden_size) > 0:
            rnn_sizes = list(gru_hidden_size)
        else:
            raise TypeError("gru_hidden_size must be an int or a nonempty list/tuple of ints")

        ### Temporal Encoder (GRU) layers architecture ###
        self.rnn_layers = nn.ModuleList()
        in_dim = input_size
        for h in rnn_sizes:
            self.rnn_layers.append(nn.GRU(in_dim, h, batch_first=True))
            in_dim = h
        self.rnn_output_size = rnn_sizes[-1]

        # Static softmax pooling parameters initialization
        if self.pooling == "static_softmax":
            self.lag_scores = nn.Parameter(torch.zeros(self.sequence_length))

        # MLP Decoder width (accept int or list/tuple)
        if isinstance(mlp_hidden_size, int):
            mlp_sizes = [mlp_hidden_size]
        elif isinstance(mlp_hidden_size, (list, tuple)) and len(mlp_hidden_size) > 0:
            mlp_sizes = list(mlp_hidden_size)
        else:
            raise TypeError("mlp_hidden_size must be an int or a nonempty list/tuple of ints")

        ### MLP Decoder layers architecture ###
        self.mlp_layers = nn.ModuleList()
        for i in range(len(mlp_sizes)):
            in_dim = self.rnn_output_size if i == 0 else mlp_sizes[i - 1]
            out_dim = mlp_sizes[i]
            self.mlp_layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU()
            ))

        ### Final linear layer ###
        self.output = nn.Linear(mlp_sizes[-1], output_size)

    
    ### Forward pass ###
    ####################

    def forward(self, x):

        ### Temporal Encoder (GRU) ###
        for rnn in self.rnn_layers:
            x, h_n = rnn(x)  
        seq_out = x  

        ### Pooling Layer ###
        if self.pooling == "last":
            h_pooled = h_n[-1]  
        else:  # static_softmax
            weights = torch.softmax(self.lag_scores.to(x.device), dim=0).view(1, -1, 1)  
            h_pooled = (seq_out * weights).sum(dim=1)  

        ### MLP Decoder ###
        z = h_pooled
        for layer in self.mlp_layers:
            z = layer(z)

        ### Final linear layer ###
        return h_pooled, self.output(z)

    

#################################################################################################################
                                            # MLP
#################################################################################################################


class MLP(nn.Module):
    """
    Multi Layer Perceptron for deterministic regression.

    Args:
        input_size (int): Number of input features.
        mlp_hidden_size (int or list/tuple[int]): Hidden units in each hidden layer.
        output_size (int): Dimensionality of the output.
    """

    def __init__(self, input_size, mlp_hidden_size, output_size):
        super().__init__()
        layers = []
        in_dim = input_size

        # Normalize layer widths (accept int or list/tuple)
        if isinstance(mlp_hidden_size, int):
            hidden_dims = [mlp_hidden_size]
        elif isinstance(mlp_hidden_size, (list, tuple)) and len(mlp_hidden_size) > 0:
            hidden_dims = list(mlp_hidden_size)
        else:
            raise TypeError("mlp_hidden_size must be an int or a nonempty list/tuple of ints.")

        ### MLP layers architecture ###
        for h in hidden_dims:
            if not isinstance(h, int) or h <= 0:
                raise ValueError("All hidden layer sizes must be positive integers.")
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.hidden_layers = nn.Sequential(*layers)

        ### Final linear layer ###
        self.output_layer = nn.Linear(in_dim, output_size)

    ### Forward pass ###
    ####################

    def forward(self, x):
        last_hidden = self.hidden_layers(x)
        return last_hidden, self.output_layer(last_hidden)

#################################################################################################################