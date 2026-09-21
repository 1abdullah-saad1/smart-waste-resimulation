from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    Set deterministic random seeds for Python, NumPy and PyTorch.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_rng(seed: int) -> np.random.Generator:
    """
    Create an independent NumPy random generator.
    """

    return np.random.default_rng(seed)