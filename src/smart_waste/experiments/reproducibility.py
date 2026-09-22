from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(
    master_seed: int,
    namespace: str,
) -> int:
    """Derive a deterministic 32-bit seed for one subsystem."""

    if master_seed < 0:
        raise ValueError("master_seed must be non-negative")

    if not namespace.strip():
        raise ValueError("namespace cannot be empty")

    payload = f"{master_seed}:{namespace}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="big",
        signed=False,
    )


def create_rng(
    master_seed: int,
    namespace: str,
) -> np.random.Generator:
    """Create an independent deterministic RNG stream."""

    return np.random.default_rng(
        derive_seed(
            master_seed,
            namespace,
        )
    )
