from smart_waste.core.city import City
from smart_waste.core.distance import euclidean_distance
from smart_waste.routing.tsr import (
    assign_bins_to_static_sectors,
    nearest_neighbor_order,
    simulate_tsr,
)


MASTER_SEED = 20260922

city = City.generate(
    area_km2=50.0,
    num_bins=1000,
    master_seed=MASTER_SEED,
)

assignments = assign_bins_to_static_sectors(
    city,
    num_trucks=10,
)

# --------------------------------------------------
# Route-only distance:
# ignore truck capacity for diagnostic purposes only.
# --------------------------------------------------

route_only_distance = 0.0

for assigned_bins in assignments:
    ordered = nearest_neighbor_order(
        city,
        assigned_bins,
    )

    current = city.depot_location

    for bin_id in ordered:
        destination = city.bins[bin_id].location

        route_only_distance += euclidean_distance(
            current,
            destination,
        )

        current = destination

    route_only_distance += euclidean_distance(
        current,
        city.depot_location,
    )


# --------------------------------------------------
# Actual capacity-constrained TSR
# --------------------------------------------------

result = simulate_tsr(
    city,
    num_trucks=10,
    truck_capacity_tonnes=10.0,
    fuel_efficiency_km_per_litre=2.5,
    bin_volume_m3=1.1,
    waste_density_kg_per_m3=229.088,
)

capacity_extra_distance = (
    result.total_distance_km
    - route_only_distance
)

route_only_fuel = (
    route_only_distance / 2.5
)

capacity_extra_fuel = (
    capacity_extra_distance / 2.5
)


print("=" * 50)
print("TSR DIAGNOSTIC")
print("=" * 50)

print(
    f"Route-only distance : "
    f"{route_only_distance:.3f} km"
)

print(
    f"Route-only fuel     : "
    f"{route_only_fuel:.3f} L"
)

print(
    f"Capacity TSR distance: "
    f"{result.total_distance_km:.3f} km"
)

print(
    f"Capacity TSR fuel    : "
    f"{result.total_fuel_litres:.3f} L"
)

print(
    f"Extra distance caused by capacity: "
    f"{capacity_extra_distance:.3f} km"
)

print(
    f"Extra fuel caused by capacity: "
    f"{capacity_extra_fuel:.3f} L"
)

print(
    f"Depot returns: "
    f"{result.total_depot_returns}"
)

print(
    f"Collected mass: "
    f"{result.total_collected_mass_tonnes:.3f} t"
)

print("=" * 50)