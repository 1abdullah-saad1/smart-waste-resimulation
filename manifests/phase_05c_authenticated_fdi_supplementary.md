# Phase 05C — Authenticated Compromised-Node FDI

## Status
SUPPLEMENTARY EXPERIMENT — not part of the primary manuscript reproduction.

## Purpose
Evaluate the security boundary of PoA/signature verification when a
legitimate edge node is compromised and the attacker possesses valid
signing credentials.

## Results

| Attack Rate | Unprotected Fuel | PoA Fuel | Fuel Saved by PoA |
|---|---:|---:|---:|
| 0%  | 50.672 L | 50.672 L | 0.000 L |
| 10% | 56.828 L | 56.828 L | 0.000 L |
| 15% | 59.732 L | 59.732 L | 0.000 L |
| 20% | 62.402 L | 62.402 L | 0.000 L |
| 30% | 67.800 L | 67.800 L | 0.000 L |

## Interpretation
PoA/signature verification rejects unauthenticated forged telemetry,
but it does not independently validate the semantic truthfulness of a
measurement produced by an authenticated compromised node.

This experiment extends the manuscript threat model and does not
replace the manuscript's primary unauthenticated-FDI scenario.