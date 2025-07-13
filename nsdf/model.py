# model.py --------------------------------------------------------------------
import math, torch, torch.nn as nn
from typing import Optional

class FourierEmbedding(nn.Module):
    """
    Positional encoding a la NeRF: for each input dimension x,
    returns [sin(2^0 πx), cos(2^0 πx), …, sin(2^{L-1} πx), cos(2^{L-1} πx)].
    """
    def __init__(self, in_features: int = 3, num_frequencies: int = 6):
        super().__init__()
        self.in_features     = in_features
        self.num_frequencies = num_frequencies
        # (L,) tensor of 1, 2, 4, … 2^{L-1}
        self.register_buffer("freq_bands",
                             2.0 ** torch.arange(num_frequencies).float() * math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (..., in_features)
        returns (..., in_features * 2 * num_frequencies)
        """
        # (..., 1, in_features) * (L,) broadcast → (..., L, in_features)
        x_exp = x.unsqueeze(-2) * self.freq_bands.view(-1, 1)
        sin, cos = torch.sin(x_exp), torch.cos(x_exp)
        embed = torch.cat([sin, cos], dim=-2)        # (..., 2L, in_features)
        embed = embed.flatten(-2)                    # (..., 2L*in_features)
        return embed

class SDFMLP(nn.Module):
    """
    Fourier‑feature MLP for signed‑distance regression.
    """
    def __init__(self,
                 in_features: int = 3,
                 num_frequencies: int = 6,
                 hidden: int = 128,
                 depth: int = 4):
        super().__init__()
        self.embed = FourierEmbedding(in_features, num_frequencies)
        input_dim  = in_features * num_frequencies * 2
        layers = [nn.Linear(input_dim, hidden), nn.SiLU(inplace=True)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU(inplace=True)]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3) – assume coordinates are already scaled to the training bbox
        returns (B,) signed distance
        """
        return self.net(self.embed(x)).squeeze(-1)
