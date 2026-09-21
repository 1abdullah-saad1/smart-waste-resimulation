from smart_waste.attacks.fdi import (
    generate_fdi_attack,
)
from smart_waste.core.city import City
from smart_waste.routing.hdr import simulate_hdr


MASTER_SEED = 20260922

ATTACK_RATES = [
    0.00,
    0.10,
    0.15,
    0.20,
    0.30,
]


city = City.generate(
    area_km2=50.0,
    num_bins=1000,
    master_seed=MASTER_SEED,
)


def run_hdr(reported=None):
    return simulate_hdr(
        city,
        num_trucks=10,
        truck_capacity_tonnes=10.0,
        fuel_efficiency_km_per_litre=2.5,
        bin_volume_m3=1.1,
        waste_density_kg_per_m3=229.088,
        threshold_percent=80.0,
        reported_fill_percent=reported,
    )


nominal = run_hdr()


print("=" * 92)
print("PHASE 05 — EXTERNAL UNAUTHENTICATED FDI — SMOKE TEST")
print("=" * 92)

print(
    f"{'Rate':>6} "
    f"{'Attacked':>9} "
    f"{'False alerts':>13} "
    f"{'HDR bins':>9} "
    f"{'Unprotected L':>14} "
    f"{'PoA bins':>9} "
    f"{'PoA L':>10}"
)

print("-" * 92)


for rate in ATTACK_RATES:

    scenario = generate_fdi_attack(
        city,
        master_seed=MASTER_SEED,
        attack_rate=rate,
        attack_type="external_unauthenticated",
        forged_fill_percent=100.0,
        threshold_percent=80.0,
        replicate_id=0,
    )

    unprotected = run_hdr(
        scenario.reported_fill_overrides()
    )

    protected = run_hdr(
        scenario.poa_accepted_overrides()
    )

    print(
        f"{rate:6.0%} "
        f"{scenario.num_attacked_nodes:9d} "
        f"{len(scenario.false_service_alert_ids(80.0)):13d} "
        f"{unprotected.total_bins_visited:9d} "
        f"{unprotected.total_fuel_litres:14.3f} "
        f"{protected.total_bins_visited:9d} "
        f"{protected.total_fuel_litres:10.3f}"
    )


print("-" * 92)

print(
    f"Nominal HDR fuel: "
    f"{nominal.total_fuel_litres:.3f} L"
)

print("=" * 92)