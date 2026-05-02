"""SR-FCA: Successive Refine Federated Clustering Algorithm.

Reference
---------
Vardhan, Ghosh, Mazumdar. "An Improved Federated Clustering Algorithm
with Model-based Clustering." TMLR, 02/2024 (``paper/paper.pdf``).

Subroutines (Algorithms 1-4 in the paper):

- :func:`one_shot`        : Algorithm 1, lines 1-4 -- pairwise-distance
                             graph -> connected components of size >= t.
- :func:`recluster`       : Algorithm 2 -- assign each client to the
                             cluster whose model is closest.
- :func:`merge`           : Algorithm 4 -- merge clusters whose models
                             are within lambda.
- :func:`refine_step`     : one REFINE round =
                             TrimmedMeanGD -> RECLUSTER -> MERGE.
- :func:`sr_fca`          : outer loop = ONE_SHOT + R REFINE steps.

Pluggable distances:

- :func:`l2_dist_model_model` -- exact L2 on flat weights; used by the
  synthetic smoke test and anywhere Assumptions 4.3-4.5 (strong
  convexity + smoothness + Lipschitz) hold.
- :func:`cross_cluster_loss_model_model` -- paper's Definition 5.1,
  ``0.5 * (f_i(w_j) + f_j(w_i))``, used for neural nets on MNIST /
  Fashion-MNIST where weights live on a non-trivial loss surface.

Running ``python -m src.srfca --smoke-test`` executes the synthetic
2-Gaussian sanity check from the plan (Phase 1 verification).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable, Iterable

import networkx as nx
import numpy as np
import torch

from .datasets import ClientData, FederatedData, make_gaussian_clusters
from .federated import (
    TrainConfig,
    local_train,
    loss_on_client,
    train_cluster_model,
    trimmed_mean,
)
from .metrics import misclustering_error
from .models import build_model
from .utils import get_device, set_seed

# -------------------- distance functions -------------------- #


def l2_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    """Euclidean distance between two flat parameter vectors."""
    return float(torch.linalg.norm(a - b).item())


def l2_pairwise(flats: list[torch.Tensor]) -> np.ndarray:
    """Full m-by-m L2 distance matrix over client weight vectors."""
    stacked = torch.stack(flats, dim=0)
    d = torch.cdist(stacked, stacked, p=2)
    return d.cpu().numpy()


def cross_cluster_loss_pairwise(
    flats: list[torch.Tensor],
    clients: list[ClientData],
    model_factory,
    task: str,
    device: torch.device,
) -> np.ndarray:
    """Paper's cross-cluster loss distance, eq. 5.1.

    ``dist(i, j) = 0.5 * (f_i(w_j) + f_j(w_i))``.

    Diagonal is zero by construction. Note we build one model instance
    and swap weights into it rather than instantiating ``m`` models --
    keeps memory flat at ``O(P)`` instead of ``O(mP)``.
    """
    m = len(flats)
    model = model_factory().to(device)
    # Pre-compute loss(w_j, data_i) for all (i, j).
    L = np.zeros((m, m), dtype=np.float32)
    for j in range(m):
        model.load_flat_weights(flats[j].to(device))  # type: ignore[attr-defined]
        for i in range(m):
            L[i, j] = loss_on_client(model, clients[i], task, device)
    return 0.5 * (L + L.T)


# -------------------- Algorithm 1: ONE_SHOT -------------------- #


@dataclass
class Clustering:
    """A clustering assignment over ``m`` clients.

    ``assignments[i]`` = cluster id of client i, or ``-1`` if client i
    is unclustered (e.g., fell into a component smaller than ``t``).
    Cluster ids are contiguous non-negative integers starting at 0.
    """

    assignments: np.ndarray  # shape [m], dtype int64
    cluster_models: dict[int, torch.Tensor] = field(default_factory=dict)

    def clusters(self) -> dict[int, list[int]]:
        """Cluster id -> list of client indices belonging to it."""
        out: dict[int, list[int]] = {}
        for i, c in enumerate(self.assignments):
            if c < 0:
                continue
            out.setdefault(int(c), []).append(i)
        return out

    def n_clusters(self) -> int:
        return len(self.clusters())


def one_shot(
    flats: list[torch.Tensor],
    lambda_: float,
    t: int,
    dist_matrix: np.ndarray | None = None,
) -> Clustering:
    """Algorithm 1 lines 1-4: initial clustering from pairwise distances.

    Build an undirected graph with ``m`` nodes and an edge ``(i, j)``
    whenever ``dist(w_i, w_j) <= lambda_``. Return the connected
    components of size ``>= t``; smaller components are marked
    unclustered (assignment ``-1``).

    Parameters
    ----------
    flats : list of torch.Tensor
        Per-client flat weights after local training.
    lambda_ : float
        Distance threshold.
    t : int
        Minimum cluster size.
    dist_matrix : np.ndarray, optional
        Precomputed ``m x m`` distance matrix. If ``None``, uses L2.
    """
    m = len(flats)
    if dist_matrix is None:
        dist_matrix = l2_pairwise(flats)

    g = nx.Graph()
    g.add_nodes_from(range(m))
    iu, ju = np.triu_indices(m, k=1)
    mask = dist_matrix[iu, ju] <= lambda_
    g.add_edges_from(zip(iu[mask].tolist(), ju[mask].tolist()))

    assignments = np.full(m, -1, dtype=np.int64)
    cid = 0
    for comp in nx.connected_components(g):
        if len(comp) >= t:
            for i in comp:
                assignments[i] = cid
            cid += 1
    return Clustering(assignments=assignments)


# -------------------- Algorithm 2: RECLUSTER -------------------- #


def recluster(
    cluster_models: dict[int, torch.Tensor],
    client_flats: list[torch.Tensor],
    clients: list[ClientData] | None = None,
    dist: str = "l2",
    model_factory=None,
    task: str = "classification",
    device: torch.device | None = None,
) -> Clustering:
    """Algorithm 2: reassign every client to its nearest cluster model.

    ``dist`` controls which distance is minimised:
    - ``"l2"``             : L2 distance on flat weights.
    - ``"cross_cluster"``  : paper's cross-cluster loss (eq. 5.1); needs
                               ``clients``, ``model_factory``, ``task``,
                               ``device``.
    """
    m = len(client_flats)
    cids = sorted(cluster_models.keys())
    if dist == "l2":
        d_mat = np.zeros((m, len(cids)), dtype=np.float32)
        for jj, c in enumerate(cids):
            for i in range(m):
                d_mat[i, jj] = l2_dist(client_flats[i], cluster_models[c])
    elif dist == "cross_cluster":
        assert clients is not None and model_factory is not None and device is not None
        model = model_factory().to(device)
        # f_i(w_c):
        loss_ic = np.zeros((m, len(cids)), dtype=np.float32)
        for jj, c in enumerate(cids):
            model.load_flat_weights(cluster_models[c].to(device))  # type: ignore[attr-defined]
            for i in range(m):
                loss_ic[i, jj] = loss_on_client(model, clients[i], task, device)
        # f_c(w_i) - we treat the cluster's "client loss" as the mean of
        # its member clients; but for RECLUSTER the paper simply uses
        # dist(w_i, w_c) = f_i(w_c) since the cluster model's loss on
        # every client is symmetric under the cross-cluster definition
        # (Section 5). Using f_i(w_c) alone is the standard
        # simplification and matches the author implementation.
        d_mat = loss_ic
    else:
        raise ValueError(f"unknown dist: {dist!r}")

    new_assign = np.array([cids[int(np.argmin(d_mat[i]))] for i in range(m)], dtype=np.int64)
    # Renumber so cluster ids are contiguous 0..K'-1.
    unique, inv = np.unique(new_assign, return_inverse=True)
    renumbered = inv.astype(np.int64)
    new_models = {
        int(new_id): cluster_models[int(old_id)] for new_id, old_id in enumerate(unique.tolist())
    }
    return Clustering(assignments=renumbered, cluster_models=new_models)


# -------------------- Algorithm 4: MERGE -------------------- #


def merge(
    clustering: Clustering,
    lambda_: float,
    t: int,
    clients: list[ClientData] | None = None,
    dist: str = "l2",
    model_factory=None,
    task: str = "classification",
    device: torch.device | None = None,
) -> Clustering:
    """Algorithm 4: merge clusters whose models are within ``lambda_``.

    Build a graph over cluster ids with edges where cluster models are
    within lambda. Each connected component becomes a single cluster;
    its merged model is the mean of the constituent cluster models
    (paper: algorithm 4 line 12).

    Clusters whose size falls below ``t`` after merging are dropped.
    """
    models = clustering.cluster_models
    cids = sorted(models.keys())
    K = len(cids)

    g = nx.Graph()
    g.add_nodes_from(cids)

    if dist == "l2":
        for ai in range(K):
            for bi in range(ai + 1, K):
                a, b = cids[ai], cids[bi]
                if l2_dist(models[a], models[b]) <= lambda_:
                    g.add_edge(a, b)
    elif dist == "cross_cluster":
        assert clients is not None and model_factory is not None and device is not None
        # cluster_loss(c, c') = mean over clients of cluster c of
        # loss under model w_{c'}, averaged symmetrically.
        clusters_members = clustering.clusters()
        model = model_factory().to(device)
        avg_loss = np.zeros((K, K), dtype=np.float32)
        for bi, b in enumerate(cids):
            model.load_flat_weights(models[b].to(device))  # type: ignore[attr-defined]
            for ai, a in enumerate(cids):
                member_idxs = clusters_members[a]
                losses = [loss_on_client(model, clients[i], task, device) for i in member_idxs]
                avg_loss[ai, bi] = float(np.mean(losses)) if losses else np.inf
        sym = 0.5 * (avg_loss + avg_loss.T)
        for ai in range(K):
            for bi in range(ai + 1, K):
                if sym[ai, bi] <= lambda_:
                    g.add_edge(cids[ai], cids[bi])
    else:
        raise ValueError(f"unknown dist: {dist!r}")

    # Assignments per component.
    new_assign = np.full_like(clustering.assignments, -1)
    new_models: dict[int, torch.Tensor] = {}
    new_id = 0
    members = clustering.clusters()
    for comp in nx.connected_components(g):
        comp_clients = [i for c in comp for i in members.get(c, [])]
        if len(comp_clients) < t:
            continue
        for i in comp_clients:
            new_assign[i] = new_id
        new_models[new_id] = torch.stack([models[c] for c in comp], dim=0).mean(dim=0)
        new_id += 1
    return Clustering(assignments=new_assign, cluster_models=new_models)


# -------------------- REFINE + outer loop -------------------- #


@dataclass
class SRFCAConfig:
    """Hyperparameters of SR-FCA.

    ``dist_mode`` chooses which distance is used by ONE_SHOT / MERGE /
    RECLUSTER. ``"l2"`` is cheap and exact for strongly-convex losses;
    ``"cross_cluster"`` matches the paper's §5 setup for neural nets.
    """

    lambda_: float = 1.0
    t: int = 2
    beta: float = 0.1
    R: int = 1                   # number of REFINE steps
    rounds_per_refine: int = 5   # T in TrimmedMeanGD
    local_train_cfg: TrainConfig = field(default_factory=TrainConfig)
    dist_mode: str = "l2"        # "l2" or "cross_cluster"


def refine_step(
    clustering: Clustering,
    clients: list[ClientData],
    cfg: SRFCAConfig,
    model_factory,
    device: torch.device,
) -> Clustering:
    """One REFINE step: TrimmedMeanGD within each cluster, then RECLUSTER, then MERGE."""
    members = clustering.clusters()
    new_models: dict[int, torch.Tensor] = {}
    for c, idxs in members.items():
        init = clustering.cluster_models.get(c)
        new_models[c] = train_cluster_model(
            model_factory=model_factory,
            clients=[clients[i] for i in idxs],
            cfg=cfg.local_train_cfg,
            device=device,
            rounds=cfg.rounds_per_refine,
            beta=cfg.beta,
            init_flat=init,
        )
    refined = Clustering(assignments=clustering.assignments.copy(), cluster_models=new_models)

    # Build current client flats for RECLUSTER's L2 branch.
    if cfg.dist_mode == "l2":
        client_flats = [new_models[int(c)] if c >= 0 else new_models[next(iter(new_models))]
                        for c in refined.assignments]
        reclustered = recluster(new_models, client_flats, dist="l2")
    else:
        reclustered = recluster(
            new_models,
            client_flats=[torch.zeros(1) for _ in clients],  # unused in cross_cluster branch
            clients=clients,
            dist="cross_cluster",
            model_factory=model_factory,
            task=cfg.local_train_cfg.task,
            device=device,
        )

    merged = merge(
        reclustered,
        lambda_=cfg.lambda_,
        t=cfg.t,
        clients=clients,
        dist=cfg.dist_mode,
        model_factory=model_factory,
        task=cfg.local_train_cfg.task,
        device=device,
    )
    return merged


def sr_fca(
    fed: FederatedData,
    cfg: SRFCAConfig,
    device: torch.device | None = None,
    verbose: bool = False,
) -> Clustering:
    """Full SR-FCA pipeline (Algorithm 1).

    Steps
    -----
    1. Each client trains locally for ``local_train_cfg.local_epochs``
       epochs starting from a shared random init.
    2. :func:`one_shot` builds the initial clustering from pairwise
       distances on those local models.
    3. :func:`refine_step` runs R times, each applying
       TrimmedMeanGD -> RECLUSTER -> MERGE.

    The returned :class:`Clustering` always has cluster models
    populated, so it can be evaluated end-to-end.
    """
    device = device or get_device()
    m = len(fed.clients)

    # Shared random init so ONE_SHOT's pairwise distances are
    # meaningful (paper Section 3.1, w_0).
    model = build_model(fed).to(device)
    w0 = model.flatten_weights().cpu()  # type: ignore[attr-defined]
    model_factory = lambda: build_model(fed)

    flats = [
        local_train(model, c, cfg.local_train_cfg, device, init_flat=w0) for c in fed.clients
    ]

    if cfg.dist_mode == "l2":
        dist_mat = l2_pairwise(flats)
    else:
        dist_mat = cross_cluster_loss_pairwise(
            flats, fed.clients, model_factory, cfg.local_train_cfg.task, device
        )

    clustering = one_shot(flats, lambda_=cfg.lambda_, t=cfg.t, dist_matrix=dist_mat)

    # Seed cluster_models by averaging the local models inside each
    # component -- gives REFINE a warm start (matches the paper's
    # implicit w_c,0 choice before TrimmedMeanGD runs).
    for cid, idxs in clustering.clusters().items():
        clustering.cluster_models[cid] = torch.stack([flats[i] for i in idxs], dim=0).mean(dim=0)

    if verbose:
        print(f"[SR-FCA] ONE_SHOT produced {clustering.n_clusters()} clusters")

    for r in range(cfg.R):
        clustering = refine_step(clustering, fed.clients, cfg, model_factory, device)
        if verbose:
            print(f"[SR-FCA] after REFINE #{r + 1}: {clustering.n_clusters()} clusters")

    return clustering


# -------------------- smoke test -------------------- #


def _smoke_test(seed: int = 0) -> int:
    """Synthetic 2-Gaussian sanity check.

    Uses :func:`make_gaussian_clusters` to build ``m=20`` clients in
    ``K=2`` well-separated clusters under a quadratic (strongly
    convex) loss. SR-FCA should recover the true clustering with zero
    misclustering error.

    Returns the integer exit code (``0`` on success, ``1`` on failure).
    """
    set_seed(seed)
    device = get_device()
    fed = make_gaussian_clusters(m=20, K=2, d=10, n_per_client=200, seed=seed)
    cfg = SRFCAConfig(
        # Empirically, intra-cluster L2 ~ 1 and inter-cluster ~ 10 on
        # this synthetic; any lambda in ~[3, 9] separates them cleanly.
        lambda_=5.0,
        t=2,
        beta=0.1,
        R=1,
        rounds_per_refine=3,
        local_train_cfg=TrainConfig(lr=0.05, local_epochs=5, batch_size=50, task="regression"),
        dist_mode="l2",
    )
    clustering = sr_fca(fed, cfg, device=device, verbose=True)
    err = misclustering_error(fed.cluster_labels(), clustering.assignments)
    print(f"[smoke-test] misclustering_error = {err:.4f}")
    print(f"[smoke-test] recovered {clustering.n_clusters()} clusters (true K={fed.n_clusters})")
    if err == 0.0 and clustering.n_clusters() == fed.n_clusters:
        print("[smoke-test] PASS ✓")
        return 0
    print("[smoke-test] FAIL ✗")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="SR-FCA runner")
    parser.add_argument("--smoke-test", action="store_true", help="run 2-Gaussian sanity check")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.smoke_test:
        return _smoke_test(seed=args.seed)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
