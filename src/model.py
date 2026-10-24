"""Tiny dense autoencoder. Trained on normal traffic only, so records it
reconstructs badly look unlike normal traffic -> anomaly."""
import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, bottleneck_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@torch.no_grad()
def per_feature_error(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Squared reconstruction error per record and per encoded feature."""
    model.eval()
    return (model(x) - x) ** 2


def reconstruction_error(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Anomaly score: mean squared reconstruction error per record."""
    return per_feature_error(model, x).mean(dim=1)
