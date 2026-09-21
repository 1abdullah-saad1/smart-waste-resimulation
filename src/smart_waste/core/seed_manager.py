from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    Set global random seeds for Python, NumPy and PyTorch.
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


def derive_seed(master_seed: int, namespace: str) -> int:
    """
    Derive a deterministic child seed from a master seed and namespace.

    Example:
        derive_seed(20260922, "city")
        derive_seed(20260922, "fills")
        derive_seed(20260922, "attack")
    """

    payload = f"{master_seed}:{namespace}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(digest[:4], byteorder="big", signed=False)