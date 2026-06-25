"""Minimal sparse autoencoder for residual-stream activations."""

from __future__ import annotations

import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, activation_dim: int, dict_size: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=True)
        self.relu = nn.ReLU()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def sae_loss(x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor, l1_coef: float = 1e-3) -> torch.Tensor:
    recon = (x - x_hat).pow(2).mean()
    sparsity = z.abs().mean()
    return recon + l1_coef * sparsity
