"""Sparse autoencoder for residual-stream activations (Bricken et al. 2023 setup)."""

from __future__ import annotations

import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """x -> ReLU(W_e (x - b_d) + b_e) -> W_d z + b_d"""

    def __init__(self, d_in: int, d_dict: int):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_dict, bias=True)
        self.decoder = nn.Linear(d_dict, d_in, bias=True)
        self.relu = nn.ReLU()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.encoder(x - self.decoder.bias))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def init_decoder_bias(sae: SparseAutoencoder, activations: torch.Tensor) -> None:
    sae.decoder.bias.data.copy_(activations.mean(dim=0))


def sae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    z: torch.Tensor,
    l1_coef: float = 1e-3,
) -> torch.Tensor:
    return (x - x_hat).pow(2).mean() + l1_coef * z.abs().mean()
