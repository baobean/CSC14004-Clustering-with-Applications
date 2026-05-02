"""Model architectures.

- ``MLP2``      : 2-layer feedforward net for MNIST / Fashion-MNIST
                   (matches paper §5, "2-layer feedforward NN").
- ``LinearRegressor``: single linear layer used by the synthetic
                   Gaussian smoke test (strongly-convex quadratic loss,
                   matching Assumption 4.3 of the paper).

Both subclass ``nn.Module`` and expose ``flatten_weights()`` /
``load_flat_weights()`` helpers because SR-FCA repeatedly compares
and aggregates flat parameter vectors. Keeping the helpers on the
model (instead of sprinkling ``state_dict`` surgery across the
codebase) keeps the rest of the code short.
"""

from __future__ import annotations

import torch
from torch import nn


class _FlatParamsMixin:
    """Mixin exposing model parameters as a single flat vector."""

    def flatten_weights(self) -> torch.Tensor:
        """Detached flat parameter vector (shape ``[P]``)."""
        return torch.cat([p.detach().reshape(-1) for p in self.parameters()])

    def load_flat_weights(self, flat: torch.Tensor) -> None:
        """Overwrite parameters in-place from a flat vector."""
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[idx : idx + n].view_as(p))
            idx += n

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MLP2(_FlatParamsMixin, nn.Module):
    """Two-layer MLP with ReLU: ``28*28 -> hidden -> n_classes``."""

    def __init__(self, input_shape: tuple[int, ...] = (28, 28), hidden: int = 200, n_classes: int = 10):
        super().__init__()
        in_dim = int(torch.tensor(input_shape).prod().item())
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearRegressor(_FlatParamsMixin, nn.Module):
    """``y = <x, w> + b`` used by the synthetic smoke test."""

    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).squeeze(-1)


def build_model(fed_data) -> nn.Module:
    """Dispatch from a :class:`FederatedData` to the right architecture.

    Classification data (``n_classes > 1``) -> :class:`MLP2`.
    Regression data    (``n_classes == 1``) -> :class:`LinearRegressor`.
    """
    if fed_data.n_classes == 1:
        d = int(torch.tensor(fed_data.input_shape).prod().item())
        return LinearRegressor(d)
    return MLP2(input_shape=fed_data.input_shape, n_classes=fed_data.n_classes)
