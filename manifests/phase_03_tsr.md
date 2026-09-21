# Phase 03 — Traditional Static Routing (TSR)

## Configuration
- Master seed: 20260922
- Area: 50 km²
- Bins: 1000
- Trucks: 10
- Truck capacity: 10 tonnes
- Fuel efficiency: 2.5 km/L
- Bin volume assumption: 1.1 m³
- Nominal waste density assumption: 229.088 kg/m³
- Geometry assumption: square region
- Distance metric: Euclidean
- Depot: city center
- Static assignment: 10 angular geographical sectors
- In-sector order: nearest neighbour

## Results
- Total waste: 126.331 tonnes
- Route-only distance: 254.011 km
- Route-only fuel: 101.604 L
- Capacity-constrained distance: 312.875 km
- Capacity-constrained fuel: 125.150 L
- Capacity-triggered depot returns: 10
- Extra distance caused by capacity: 58.864 km
- Extra fuel caused by capacity: 23.546 L

## Interpretation
The original manuscript reported approximately 100 L for TSR.
The reconstructed route-only result is close to that value, but
the capacity-constrained implementation requires additional depot
returns and therefore consumes 125.150 L.

The original implementation cannot be inferred conclusively because
the lost simulation files and manuscript do not specify the waste-mass
conversion or truck unloading/return policy.