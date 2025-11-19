#################################################################################################################
                                            # Import libraries
#################################################################################################################


import torch
import torch.nn as nn


#######################################################################################################################
                            # Conditional Noise Encoder 
#######################################################################################################################

class Conditional_Noise_Encoder(nn.Module):
    """
    Generates input-dependent Gaussian noise z ~ N(0, σ(x)^2) from the input space.

    Representations:
        - 'scalar': one variance shared across all noise dimensions.
        - 'vector': one variance per noise dimension.

    Components:
        - σ(x): linear projection with Softplus activation to ensure positivity, then clamped.

    Args:
        input_dim (int): Number of input features.
        noise_dim (int): Dimensionality of the noise vector.
        representation (str): 'scalar' or 'vector'.
        sigma_min (float): Minimum allowed standard deviation. Default is 1e-6.
        sigma_max (float): Maximum allowed standard deviation. Default is 10.0.
    """

    def __init__(self, input_dim, noise_dim, representation='scalar',
                 sigma_min=1e-4, sigma_max=10.0):

        super().__init__()

        # Argument validity (checker)
        assert representation in ['scalar', 'vector'], "mode must be 'scalar' or 'vector'"
        assert sigma_min > 0.0 and sigma_max > sigma_min, "Require 0 < sigma_min < sigma_max"

        self.scalar = (representation == 'scalar')
        self.noise_dim = noise_dim

        # Linear projection + softplus over both representations
        out_dim = 1 if self.scalar else noise_dim
        self.sigma_linear = nn.Linear(input_dim, out_dim)
        self.softplus = nn.Softplus() # default β = 1

        # store bounds as floats
        self._sigma_min = float(sigma_min)  # default 1e-4
        self._sigma_max = float(sigma_max)  # default 10.0

    def forward(self, x):
        batch_size = x.size(0)

        # σ(x) = Softplus(Wx + b) then clamp to [sigma_min, sigma_max] for stability
        sigma = self.softplus(self.sigma_linear(x))
        sigma = sigma.clamp(self._sigma_min, self._sigma_max)

        # expand in scalar mode
        if self.scalar:
            sigma = sigma.expand(-1, self.noise_dim)

        # sample from N(0, σ^2)
        eps = torch.randn(batch_size, self.noise_dim, device=x.device)
        z = sigma * eps
        return z

