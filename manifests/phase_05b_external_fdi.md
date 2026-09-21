# Phase 05B — External Unauthenticated FDI

## Experimental Design
- Master city/workload seed: 20260922
- Attack replicates: 100
- Paired/nested attack severities within each replicate
- Attack rates: 0%, 10%, 15%, 20%, 30%
- Forged fill report: 100%
- Node selection: uniform without replacement
- HDR threshold: 80%
- Primary HDR: sector-constrained nearest neighbour
- Threat type: external unauthenticated telemetry forgery

## Nominal HDR
- Fuel: 50.672 L

## Results
| Attack | Mean False Alerts | Mean Unprotected Fuel | 95% CI | Mean PoA Fuel | Mean Fuel Saved |
|---|---:|---:|---:|---:|---:|
| 0% | 0.00 | 50.672 L | [50.672, 50.672] | 50.672 L | 0.000 L |
| 10% | 81.72 | 56.975 L | [56.676, 57.274] | 50.672 L | 6.303 L |
| 15% | 122.50 | 59.786 L | [59.484, 60.087] | 50.672 L | 9.114 L |
| 20% | 163.21 | 62.382 L | [62.110, 62.655] | 50.672 L | 11.710 L |
| 30% | 244.08 | 67.877 L | [67.511, 68.243] | 50.672 L | 17.205 L |

## Interpretation
These 100 replicates quantify uncertainty caused by random attacked-node
selection while holding the physical city/workload constant.

Under the external unauthenticated FDI threat model, cryptographic
verification rejects all forged events before routing, so the protected
routing input remains identical to the nominal input.

The zero-width protected confidence interval therefore applies only to
this fixed-city/fixed-workload experiment and must not be interpreted as
zero system-wide uncertainty.

The previously reported 120 L adversarial HDR result was not reproduced
under this reconstructed protocol and will not be forced by calibration.