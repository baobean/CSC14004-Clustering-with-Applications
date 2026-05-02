"""Federated-clustering dataset construction.

Produces ``m`` clients partitioned into ``K`` heterogeneous clusters
following the recipes of Vardhan et al. (TMLR 02/2024, §5).

Two simulated families are implemented:

- **Rotated MNIST** (``split="rotated"``): each cluster is assigned a
  rotation angle; all clients in that cluster see MNIST rotated by
  that angle. Paper uses angles ``{0, 90, 180, 270}`` -> 4 clusters.
  We default to ``K=2`` (``{0, 180}``) for the core-scope run, which
  keeps the problem easy enough to validate while still exhibiting
  heterogeneity. Pass ``angles=(0, 90, 180, 270)`` to match the paper.
- **Inverted MNIST** (``split="inverted"``): one cluster sees the
  original images, the other sees ``1 - x`` (pixel-inverted).

For the synthetic smoke test we additionally expose
``make_gaussian_clusters``, which returns ``m`` tiny datasets whose
ground-truth linear model is one of ``K`` cluster centres in
``R^d``. SR-FCA on this synthetic should recover ``C*`` exactly.

Every splitter returns a ``FederatedData`` namedtuple carrying the
per-client train / test tensors and the ground-truth cluster index of
each client, plus metadata needed downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms

from .utils import project_root


@dataclass
class ClientData:
    """Data held by a single federated client."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    cluster: int  # ground-truth cluster index


@dataclass
class FederatedData:
    """Container of all clients plus metadata for an FL experiment."""

    clients: list[ClientData]
    n_clusters: int
    input_shape: tuple[int, ...]
    n_classes: int
    name: str

    def cluster_labels(self) -> np.ndarray:
        """Ground-truth cluster index per client (shape ``[m]``)."""
        return np.asarray([c.cluster for c in self.clients], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.clients)


def _load_mnist(root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load MNIST as float tensors in ``[0, 1]``.

    Returns ``(x_train, y_train, x_test, y_test)``. Images are shape
    ``[N, 28, 28]`` (no channel axis; added lazily in the model).
    """
    to_tensor = transforms.ToTensor()
    tr = datasets.MNIST(str(root), train=True, download=True, transform=to_tensor)
    te = datasets.MNIST(str(root), train=False, download=True, transform=to_tensor)
    # Pull everything into dense tensors; MNIST is small.
    x_tr = torch.stack([tr[i][0].squeeze(0) for i in range(len(tr))])
    y_tr = torch.tensor([tr[i][1] for i in range(len(tr))], dtype=torch.long)
    x_te = torch.stack([te[i][0].squeeze(0) for i in range(len(te))])
    y_te = torch.tensor([te[i][1] for i in range(len(te))], dtype=torch.long)
    return x_tr, y_tr, x_te, y_te


def _load_fashion_mnist(root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load Fashion-MNIST as float tensors in ``[0, 1]``."""
    to_tensor = transforms.ToTensor()
    tr = datasets.FashionMNIST(str(root), train=True, download=True, transform=to_tensor)
    te = datasets.FashionMNIST(str(root), train=False, download=True, transform=to_tensor)
    x_tr = torch.stack([tr[i][0].squeeze(0) for i in range(len(tr))])
    y_tr = torch.tensor([tr[i][1] for i in range(len(tr))], dtype=torch.long)
    x_te = torch.stack([te[i][0].squeeze(0) for i in range(len(te))])
    y_te = torch.tensor([te[i][1] for i in range(len(te))], dtype=torch.long)
    return x_tr, y_tr, x_te, y_te


def _apply_rotation(x: torch.Tensor, angle_deg: int) -> torch.Tensor:
    """Rotate a ``[N, H, W]`` batch by 0/90/180/270 degrees using torch rot90.

    We restrict to these four angles because ``torch.rot90`` is exact,
    whereas arbitrary-angle rotation introduces interpolation noise
    that confounds the clustering signal we are trying to measure.
    """
    assert angle_deg in (0, 90, 180, 270), "only cardinal rotations supported"
    k = angle_deg // 90
    return torch.rot90(x, k=k, dims=(-2, -1)) if k else x


def _shard_indices(
    n: int, m: int, n_per_client: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Randomly partition (without replacement if possible) ``n`` samples
    into ``m`` clients of size ``n_per_client``.

    If ``m * n_per_client > n`` we fall back to sampling *with*
    replacement -- this matches the paper's setup where simulated
    datasets are constructed by resampling MNIST.
    """
    if m * n_per_client <= n:
        perm = rng.permutation(n)
        return [perm[i * n_per_client : (i + 1) * n_per_client] for i in range(m)]
    return [rng.choice(n, size=n_per_client, replace=True) for _ in range(m)]


def make_mnist_federated(
    split: str = "rotated",
    m: int = 100,
    n_per_client: int = 600,
    angles: Sequence[int] = (0, 180),
    test_per_client: int = 200,
    seed: int = 42,
    dataset: str = "mnist",
) -> FederatedData:
    """Build a clustered-FL MNIST (or Fashion-MNIST) dataset.

    Parameters
    ----------
    split : {"rotated", "inverted"}
        Type of heterogeneity. ``rotated`` assigns each cluster a
        rotation angle from ``angles``; ``inverted`` creates 2 clusters
        (original / pixel-inverted).
    m : int
        Total number of clients.
    n_per_client : int
        Number of training samples per client (paper uses 600 for MNIST).
    angles : sequence of int
        Rotation angles in degrees; must be a subset of
        ``{0, 90, 180, 270}``. Only used when ``split="rotated"``.
        The number of clusters ``K`` equals ``len(angles)``.
    test_per_client : int
        Number of test samples per client, drawn from MNIST's test set
        and transformed with the same rotation/inversion as the client's
        training data.
    seed : int
        Controls which images land on which client.
    dataset : {"mnist", "fashion_mnist"}
        Underlying image corpus. ``fashion_mnist`` is the Phase 3
        "new dataset" evaluation and uses the identical splitting
        logic.

    Returns
    -------
    FederatedData
        Container with all ``m`` clients and metadata.
    """
    rng = np.random.default_rng(seed)
    root = project_root() / "data"
    root.mkdir(exist_ok=True)

    loader = _load_mnist if dataset == "mnist" else _load_fashion_mnist
    x_tr, y_tr, x_te, y_te = loader(root)

    if split == "rotated":
        transforms_per_cluster = [lambda x, a=a: _apply_rotation(x, a) for a in angles]
        K = len(angles)
        name = f"{dataset}_rotated_k{K}"
    elif split == "inverted":
        transforms_per_cluster = [lambda x: x, lambda x: 1.0 - x]
        K = 2
        name = f"{dataset}_inverted"
    else:
        raise ValueError(f"unknown split: {split!r}")

    # Assign clusters to clients as evenly as possible.
    clients_per_cluster = m // K
    client_clusters = np.repeat(np.arange(K), clients_per_cluster)
    # If m not divisible by K, pad the remainder with random clusters.
    if client_clusters.size < m:
        extra = rng.integers(0, K, size=m - client_clusters.size)
        client_clusters = np.concatenate([client_clusters, extra])
    rng.shuffle(client_clusters)

    train_idx = _shard_indices(x_tr.size(0), m, n_per_client, rng)
    test_idx = _shard_indices(x_te.size(0), m, test_per_client, rng)

    clients: list[ClientData] = []
    for i in range(m):
        c = int(client_clusters[i])
        fn = transforms_per_cluster[c]
        xi_tr = fn(x_tr[train_idx[i]])
        xi_te = fn(x_te[test_idx[i]])
        clients.append(
            ClientData(
                x_train=xi_tr,
                y_train=y_tr[train_idx[i]],
                x_test=xi_te,
                y_test=y_te[test_idx[i]],
                cluster=c,
            )
        )

    return FederatedData(
        clients=clients,
        n_clusters=K,
        input_shape=(28, 28),
        n_classes=10,
        name=name,
    )


def make_gaussian_clusters(
    m: int = 20,
    K: int = 2,
    d: int = 10,
    n_per_client: int = 200,
    cluster_sep: float = 5.0,
    noise_std: float = 1.0,
    seed: int = 42,
) -> FederatedData:
    """Synthetic federated regression used for the smoke test.

    Each cluster ``k`` has a ground-truth weight vector ``w_k* in R^d``
    drawn from ``N(0, cluster_sep^2 I)``. Clients assigned to cluster
    ``k`` generate data ``y = <x, w_k*> + noise``. The loss is strongly
    convex (quadratic), satisfying the paper's theoretical
    assumptions -- so SR-FCA is expected to recover the clustering
    exactly in a handful of rounds.

    Returned as a ``FederatedData`` with ``n_classes=1`` (regression
    target stored as a float in ``y_train``/``y_test``).
    """
    rng = np.random.default_rng(seed)
    clusters_per_client = np.repeat(np.arange(K), m // K)
    if clusters_per_client.size < m:
        clusters_per_client = np.concatenate(
            [clusters_per_client, rng.integers(0, K, size=m - clusters_per_client.size)]
        )
    rng.shuffle(clusters_per_client)

    w_stars = rng.normal(0.0, cluster_sep, size=(K, d)).astype(np.float32)

    clients = []
    for i in range(m):
        c = int(clusters_per_client[i])
        x_tr = rng.normal(0, 1, size=(n_per_client, d)).astype(np.float32)
        y_tr = x_tr @ w_stars[c] + rng.normal(0, noise_std, size=n_per_client).astype(np.float32)
        x_te = rng.normal(0, 1, size=(n_per_client // 2, d)).astype(np.float32)
        y_te = x_te @ w_stars[c] + rng.normal(0, noise_std, size=n_per_client // 2).astype(np.float32)
        clients.append(
            ClientData(
                x_train=torch.from_numpy(x_tr),
                y_train=torch.from_numpy(y_tr),
                x_test=torch.from_numpy(x_te),
                y_test=torch.from_numpy(y_te),
                cluster=c,
            )
        )

    fd = FederatedData(
        clients=clients,
        n_clusters=K,
        input_shape=(d,),
        n_classes=1,
        name=f"gaussian_k{K}_d{d}",
    )
    fd.w_stars = w_stars  # type: ignore[attr-defined]
    return fd


def client_loader(client: ClientData, batch_size: int) -> TensorDataset:
    """Tiny convenience: return a TensorDataset for a client's training split."""
    return TensorDataset(client.x_train, client.y_train)
