"""Global utilities: seeding, device selection, simple checkpoint I/O."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> None:
    """Seed python, numpy, and torch for reproducible experiments.

    Parameters
    ----------
    seed : int
        Master seed applied to all RNGs.
    deterministic : bool
        If True, also flip torch deterministic-algorithm flags.
        Set False for speed on large runs where tiny nondeterminism is
        acceptable.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU.

    The project targets CPU; this helper exists so a Kaggle fallback
    run on GPU needs no code change.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def project_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).resolve().parent.parent


def ensure_dir(path: os.PathLike) -> Path:
    """Create ``path`` (and parents) if missing; return as ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
