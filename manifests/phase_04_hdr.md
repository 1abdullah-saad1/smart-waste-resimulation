# Phase 04 — Heuristic Dynamic Routing (HDR)

## Configuration
- Master seed: 20260922
- Area: 50 km²
- Bins: 1000
- Trucks: 10
- Service threshold: 80%
- Truck capacity: 10 tonnes
- Fuel efficiency: 2.5 km/L
- Bin volume assumption: 1.1 m³
- Waste density assumption: 229.088 kg/m³
- Distance metric: Euclidean
- Fleet assignment assumption: same 10 angular sectors as TSR
- In-sector route policy: nearest neighbour

## Nominal Results
- Eligible bins: 186
- Waste collected: 41.939 tonnes
- Total distance: 126.680 km
- Fuel: 50.672 L
- Capacity-triggered depot returns: 0

## Comparison
- Capacity-aware TSR fuel: 125.150 L
- HDR fuel reduction vs capacity-aware TSR: 59.51%
- Route-only TSR fuel: 101.604 L
- HDR fuel reduction vs route-only TSR: 50.13%

## Interpretation
The manuscript reports 62 L for nominal HDR, but the original
fleet-assignment rule is not available. The current result corresponds
to a documented sector-constrained greedy nearest-neighbour implementation.
It must not be forced to reproduce the manuscript value.