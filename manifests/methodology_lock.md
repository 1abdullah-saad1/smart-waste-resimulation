# Methodology Lock — Primary Reproduction Protocol

The primary objective of this project is to reproduce the experimental
methodology stated in the manuscript before introducing any extensions.

## Primary Routing Configuration

- 1,000 bins
- 50 km² simulated area
- 10 collection trucks
- Truck capacity: 10 tonnes
- Fuel efficiency: 2.5 km/L
- CVRP-based routing evaluation
- 100 episodes/repetitions per attack scenario

## Primary Routing Baselines

### TSR
Traditional Static Routing visits every assigned bin.

### HDR
Heuristic Dynamic Routing uses a greedy nearest-neighbour policy
and serves bins reporting >=80% fill.

### Proposed System
PoA-verified telemetry is supplied to the RL routing engine.

## Primary FDI Threat Model

- FDI severity: 10–30% of simulated nodes
- Forged report: 100% full
- Representative attack condition includes 15% compromised bins
- Invalid/unauthenticated forged telemetry is filtered by the
  cryptographic verification layer before routing.

## Reproduction Rule

Values reported in the original manuscript are reference outcomes,
not calibration targets.

If the reconstructed implementation produces different values,
the difference must be reported and explained rather than hidden
or tuned away.

## Reconstruction Assumptions

Any detail required for implementation but not explicitly specified
in the manuscript must be identified as a reconstruction assumption.

Examples currently include:
- city geometry,
- depot location,
- physical bin volume,
- waste-density conversion,
- TSR bin-to-truck assignment,
- exact multi-truck coordination rule.

## Supplementary Experiments

The following may be evaluated, but they must not replace the
primary manuscript protocol:

- paired/nested FDI severities,
- authenticated compromised-node FDI,
- alternative fleet-assignment rules,
- global versus sector HDR,
- density sensitivity,
- geometry sensitivity,
- ablation studies,
- additional statistical robustness tests.