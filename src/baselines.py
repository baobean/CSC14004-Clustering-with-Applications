"""Baselines compared against SR-FCA (paper Tables 2-4).

Each baseline is a callable ``run(fed, cfg, device) -> Clustering``.
The returned :class:`Clustering` carries per-client cluster ids and
per-cluster model weights, matching SR-FCA's output so downstream
evaluation (``results.py``) is one code path.

Implementations
---------------
- ``run_local``      : no aggregation; every client = its own cluster.
- ``run_global``     : FedAvg with a single cluster over all clients
                        (McMahan & Ramage 2017).
- ``run_ifca``       : Iterative Federated Clustering (Ghosh et al. 2022).
- ``run_local_kmeans`` : train local models, then KMeans on flat
                        weights (Ghosh et al. 2019).

Design notes
------------
IFCA needs the number of clusters ``K`` as input -- that is the
paper's main gripe with it. We pass it via ``BaselineConfig.K``;
experiments that want to tune it should sweep externally.

For "Local" we still build a single reference model to evaluate
per-client accuracy; each client's cluster id is its own index.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.cluster import KMeans

from .datasets import ClientData, FederatedData
from .federated import TrainConfig, local_train, train_cluster_model
from .models import build_model
from .srfca import Clustering
from .utils import get_device


@dataclass
class BaselineConfig:
    """Hyperparameters shared by baselines.

    ``K``        : assumed number of clusters (used by IFCA / LocalKMeans).
    ``rounds``   : number of server rounds for federated baselines.
    ``local_train_cfg`` : per-client SGD configuration.
    """

    K: int = 2
    rounds: int = 10
    local_train_cfg: TrainConfig = field(default_factory=TrainConfig)


# -------------------- Local (no FL) -------------------- #


def run_local(fed: FederatedData, cfg: BaselineConfig, device: torch.device | None = None) -> Clustering:
    """Every client trains alone -- zero inter-client collaboration.

    We return a clustering with ``m`` singleton clusters (one per
    client). This matches the paper's definition of the "Local"
    baseline in §5 and lets us reuse the same evaluation harness.
    """
    device = device or get_device()
    model = build_model(fed).to(device)
    init = model.flatten_weights().cpu()  # type: ignore[attr-defined]

    cluster_models: dict[int, torch.Tensor] = {}
    for i, c in enumerate(fed.clients):
        w = local_train(model, c, cfg.local_train_cfg, device, init_flat=init)
        cluster_models[i] = w
    assignments = np.arange(len(fed.clients), dtype=np.int64)
    return Clustering(assignments=assignments, cluster_models=cluster_models)


# -------------------- Global (FedAvg) -------------------- #


def run_global(fed: FederatedData, cfg: BaselineConfig, device: torch.device | None = None) -> Clustering:
    """Single FedAvg model across all clients (one cluster).

    Implementation: call :func:`train_cluster_model` with ``beta=0``
    (plain mean) on the full client roster for ``cfg.rounds`` rounds.
    """
    device = device or get_device()
    w = train_cluster_model(
        model_factory=lambda: build_model(fed),
        clients=fed.clients,
        cfg=cfg.local_train_cfg,
        device=device,
        rounds=cfg.rounds,
        beta=0.0,
    )
    assignments = np.zeros(len(fed.clients), dtype=np.int64)
    return Clustering(assignments=assignments, cluster_models={0: w})


# -------------------- IFCA -------------------- #


def run_ifca(fed: FederatedData, cfg: BaselineConfig, device: torch.device | None = None) -> Clustering:
    """Iterative Federated Clustering Algorithm (Ghosh et al. 2022).

    Maintains ``K`` global models. Each round, every client picks the
    model that minimises its local loss (hard assignment) and trains
    it one local epoch. The server then averages updates per cluster.

    Unlike SR-FCA, IFCA requires ``K`` and is sensitive to init --
    both documented weaknesses we will exploit in the report (§3.2).
    """
    device = device or get_device()
    rng = np.random.default_rng(0)
    K = cfg.K
    model_factory = lambda: build_model(fed)
    probe = model_factory().to(device)

    # K independent random inits.
    cluster_models = {
        k: model_factory().flatten_weights().cpu()  # type: ignore[attr-defined]
        for k in range(K)
    }

    assignments = rng.integers(0, K, size=len(fed.clients)).astype(np.int64)

    loss_fn = (
        torch.nn.CrossEntropyLoss()
        if cfg.local_train_cfg.task == "classification"
        else torch.nn.MSELoss()
    )

    for _ in range(cfg.rounds):
        # Each client picks its best cluster model (lowest training loss).
        new_assign = np.empty_like(assignments)
        with torch.no_grad():
            probe.eval()
            for i, client in enumerate(fed.clients):
                x, y = client.x_train.to(device), client.y_train.to(device)
                best_k, best_loss = 0, float("inf")
                for k in range(K):
                    probe.load_flat_weights(cluster_models[k].to(device))  # type: ignore[attr-defined]
                    loss = float(loss_fn(probe(x), y).item())
                    if loss < best_loss:
                        best_loss, best_k = loss, k
                new_assign[i] = best_k
        assignments = new_assign

        # Aggregate per cluster using FedAvg.
        for k in range(K):
            idxs = np.where(assignments == k)[0]
            if idxs.size == 0:
                continue  # keep stale model; next round may reattract clients
            flats = [
                local_train(probe, fed.clients[i], cfg.local_train_cfg, device, init_flat=cluster_models[k])
                for i in idxs
            ]
            cluster_models[k] = torch.stack(flats, dim=0).mean(dim=0)

    # Renumber so cluster ids are contiguous (drops empty clusters).
    used = sorted({int(c) for c in assignments})
    remap = {old: new for new, old in enumerate(used)}
    assignments = np.array([remap[int(c)] for c in assignments], dtype=np.int64)
    cluster_models = {remap[k]: cluster_models[k] for k in used}
    return Clustering(assignments=assignments, cluster_models=cluster_models)


# -------------------- Local-KMeans -------------------- #


def run_local_kmeans(
    fed: FederatedData, cfg: BaselineConfig, device: torch.device | None = None
) -> Clustering:
    """Train local models, then KMeans on their flat weights.

    This is the Ghosh et al. (2019) baseline. It is known to behave
    poorly on CIFAR-10 and real datasets (paper notes <=10% accuracy
    on simulated CIFAR); we include it because the paper does.

    After KMeans finds ``K`` clusters, the cluster model is the
    weighted mean of the constituent client models (a one-round FedAvg
    on the assignment, which gives a reasonable model to evaluate).
    """
    device = device or get_device()
    model = build_model(fed).to(device)
    init = model.flatten_weights().cpu()  # type: ignore[attr-defined]
    flats = [
        local_train(model, c, cfg.local_train_cfg, device, init_flat=init) for c in fed.clients
    ]
    mat = torch.stack(flats, dim=0).numpy()
    labels = KMeans(n_clusters=cfg.K, n_init=10, random_state=0).fit_predict(mat)
    assignments = labels.astype(np.int64)

    cluster_models: dict[int, torch.Tensor] = {}
    for k in range(cfg.K):
        idxs = np.where(assignments == k)[0]
        if idxs.size == 0:
            continue
        cluster_models[int(k)] = torch.stack([flats[i] for i in idxs], dim=0).mean(dim=0)
    return Clustering(assignments=assignments, cluster_models=cluster_models)


BASELINES = {
    "local": run_local,
    "global": run_global,
    "ifca": run_ifca,
    "local_kmeans": run_local_kmeans,
}
