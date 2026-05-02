"""Federated-training primitives shared by SR-FCA and all baselines.

The paper's methods differ on the server side (FedAvg vs. TrimmedMean
vs. "no aggregation"), but every client runs the same local-SGD
inner loop. Centralising these primitives lets ``srfca.py`` and
``baselines.py`` stay focused on what is distinctive to each.

Exposed functions
-----------------
- ``local_train(model, data, ...)`` -- run ``E`` epochs of SGD on a
  single client, return the resulting flat parameter vector.
- ``fedavg(flats, weights=None)`` -- uniform or weighted mean of flat
  parameter vectors (``Global`` baseline, inner loop of Local-KMeans).
- ``trimmed_mean(flats, beta)`` -- coordinate-wise trimmed mean
  (Yin et al. 2018), the robust aggregator used inside
  ``TrimmedMeanGD`` for SR-FCA's REFINE step.
- ``train_cluster_model(...)`` -- ``T`` rounds of trimmed-mean FedAvg
  over the clients of one cluster, producing the cluster model.
- ``evaluate(model, x, y, task)`` -- accuracy for classification,
  mean-squared error for regression (used by the Gaussian smoke test).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .datasets import ClientData


@dataclass
class TrainConfig:
    """Hyperparameters for ``local_train`` and ``train_cluster_model``."""

    lr: float = 0.01
    local_epochs: int = 1      # E in the paper
    batch_size: int = 64
    weight_decay: float = 0.0
    task: str = "classification"  # or "regression"


def _loss_fn(task: str) -> nn.Module:
    return nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()


def local_train(
    model: nn.Module,
    data: ClientData,
    cfg: TrainConfig,
    device: torch.device,
    init_flat: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run local SGD on one client.

    Parameters
    ----------
    model : nn.Module
        Architecture (parameters are overwritten in-place).
    data : ClientData
        Client's training tensors.
    cfg : TrainConfig
        Hyperparameters.
    device : torch.device
        Computation device.
    init_flat : torch.Tensor, optional
        Initial flat weights. If ``None`` the model's current weights
        are used (useful for ``Local`` baseline which wants random init).

    Returns
    -------
    torch.Tensor
        Flat parameter vector after local training (``shape [P]``), on CPU.
    """
    model.to(device)
    if init_flat is not None:
        model.load_flat_weights(init_flat.to(device))  # type: ignore[attr-defined]
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = _loss_fn(cfg.task)
    ds = TensorDataset(data.x_train, data.y_train)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    for _ in range(cfg.local_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
    return model.flatten_weights().cpu()  # type: ignore[attr-defined]


def fedavg(flats: list[torch.Tensor], weights: list[float] | None = None) -> torch.Tensor:
    """Uniform or weighted mean of client parameter vectors.

    Parameters
    ----------
    flats : list of torch.Tensor
        Per-client flat weight vectors (all same length).
    weights : list of float, optional
        Convex combination coefficients. ``None`` -> uniform.
    """
    stacked = torch.stack(flats, dim=0)
    if weights is None:
        return stacked.mean(dim=0)
    w = torch.tensor(weights, dtype=stacked.dtype).view(-1, 1)
    w = w / w.sum()
    return (stacked * w).sum(dim=0)


def trimmed_mean(flats: list[torch.Tensor], beta: float) -> torch.Tensor:
    """Coordinate-wise trimmed mean (Yin et al. 2018).

    For each coordinate, drop the smallest and largest
    ``floor(beta * n)`` values, then average the rest.

    Parameters
    ----------
    flats : list of torch.Tensor
        Per-client flat weight vectors (all same length).
    beta : float
        Trim fraction per side, ``0 <= beta < 0.5``. ``beta=0`` recovers
        the plain FedAvg; ``beta -> 0.5`` keeps only the coordinate-wise
        median.
    """
    if not 0 <= beta < 0.5:
        raise ValueError(f"beta must be in [0, 0.5), got {beta}")
    stacked = torch.stack(flats, dim=0)  # [n, P]
    n = stacked.shape[0]
    k = int(np.floor(beta * n))
    if k == 0:
        return stacked.mean(dim=0)
    sorted_, _ = torch.sort(stacked, dim=0)
    trimmed = sorted_[k : n - k]
    return trimmed.mean(dim=0)


def train_cluster_model(
    model_factory,
    clients: list[ClientData],
    cfg: TrainConfig,
    device: torch.device,
    rounds: int,
    beta: float,
    init_flat: torch.Tensor | None = None,
) -> torch.Tensor:
    """``TrimmedMeanGD`` subroutine from Algorithm 3 of the paper.

    For ``rounds`` rounds, every client in ``clients`` runs
    ``local_train`` from the current global weights and the server
    aggregates with :func:`trimmed_mean` at trim level ``beta``.

    Setting ``beta=0`` recovers plain FedAvg (used by the ``Global``
    baseline and by ablation A2). Setting ``rounds=0`` returns
    ``init_flat`` unchanged (useful for re-using ONE_SHOT weights).

    Returns
    -------
    torch.Tensor
        Trained cluster-model weights (flat).
    """
    if len(clients) == 0:
        raise ValueError("cannot train a cluster with zero clients")

    model = model_factory().to(device)
    if init_flat is None:
        init_flat = model.flatten_weights().cpu()  # type: ignore[attr-defined]

    w = init_flat.clone()
    for _ in range(rounds):
        flats = [local_train(model, c, cfg, device, init_flat=w) for c in clients]
        w = trimmed_mean(flats, beta=beta)
    return w


@torch.no_grad()
def evaluate(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, task: str, device: torch.device
) -> float:
    """Test-set metric for one client under one model.

    Returns accuracy (in ``[0, 1]``) for classification and *negative*
    MSE for regression (so that higher is always better in downstream
    reductions, like ``argmax`` over cluster models).
    """
    model.to(device).eval()
    x, y = x.to(device), y.to(device)
    out = model(x)
    if task == "classification":
        return float((out.argmax(dim=-1) == y).float().mean().item())
    return float(-torch.nn.functional.mse_loss(out, y).item())


@torch.no_grad()
def loss_on_client(
    model: nn.Module, data: ClientData, task: str, device: torch.device
) -> float:
    """Average training loss of ``model`` on ``data``.

    Used to compute the paper's *cross-cluster loss* distance
    (eq. 5.1). Intentionally runs on the training split -- the
    distance is a model-quality signal, not a generalisation estimate.
    """
    model.to(device).eval()
    loss_fn = _loss_fn(task)
    x, y = data.x_train.to(device), data.y_train.to(device)
    return float(loss_fn(model(x), y).item())
