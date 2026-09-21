from __future__ import annotations

from pathlib import Path

import pandas as pd

from smart_waste.attacks.fdi import (
    generate_fdi_attack,
)
from smart_waste.core.city import City
from smart_waste.routing.hdr import simulate_hdr
from smart_waste.statistics.descriptive import (
    summarize,
)


MASTER_SEED = 20260922

ATTACK_RATES = [
    0.00,
    0.10,
    0.15,
    0.20,
    0.30,
]

NUM_REPLICATES = 100

THRESHOLD_PERCENT = 80.0

OUTPUT_RAW = Path(
    "results/raw/fdi_authenticated_100_replicates.csv"
)

OUTPUT_SUMMARY = Path(
    "results/processed/fdi_authenticated_summary.csv"
)



def run_hdr(
    city: City,
    reported_fill_percent=None,
):
    return simulate_hdr(
        city,
        num_trucks=10,
        truck_capacity_tonnes=10.0,
        fuel_efficiency_km_per_litre=2.5,
        bin_volume_m3=1.1,
        waste_density_kg_per_m3=229.088,
        threshold_percent=THRESHOLD_PERCENT,
        reported_fill_percent=reported_fill_percent,
    )


def main() -> None:
    OUTPUT_RAW.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    city = City.generate(
        area_km2=50.0,
        num_bins=1000,
        master_seed=MASTER_SEED,
    )

    nominal = run_hdr(city)

    records = []

    for replicate_id in range(NUM_REPLICATES):

        for attack_rate in ATTACK_RATES:

            scenario = generate_fdi_attack(
                city,
                master_seed=MASTER_SEED,
                attack_rate=attack_rate,
                attack_type=("authenticated_compromise"),
                forged_fill_percent=100.0,
                threshold_percent=THRESHOLD_PERCENT,
                replicate_id=replicate_id,
            )

            unprotected = run_hdr(
                city,
                scenario.reported_fill_overrides(),
            )

            protected = run_hdr(
                city,
                scenario.poa_accepted_overrides(),
            )

            false_alerts = len(
                scenario.false_service_alert_ids(
                    THRESHOLD_PERCENT
                )
            )

            records.append(
                {
                    "master_seed": MASTER_SEED,
                    "replicate_id": replicate_id,
                    "selection_seed": (
                        scenario.selection_seed
                    ),
                    "attack_rate": attack_rate,
                    "attacked_nodes": (
                        scenario.num_attacked_nodes
                    ),
                    "false_service_alerts": (
                        false_alerts
                    ),
                    "unprotected_bins_visited": (
                        unprotected.total_bins_visited
                    ),
                    "unprotected_distance_km": (
                        unprotected.total_distance_km
                    ),
                    "unprotected_fuel_litres": (
                        unprotected.total_fuel_litres
                    ),
                    "protected_bins_visited": (
                        protected.total_bins_visited
                    ),
                    "protected_distance_km": (
                        protected.total_distance_km
                    ),
                    "protected_fuel_litres": (
                        protected.total_fuel_litres
                    ),
                    "nominal_fuel_litres": (
                        nominal.total_fuel_litres
                    ),
                    "fuel_overhead_unprotected_l": (
                        unprotected.total_fuel_litres
                        - nominal.total_fuel_litres
                    ),
                    "fuel_saved_by_poa_l": (
                        unprotected.total_fuel_litres
                        - protected.total_fuel_litres
                    ),
                }
            )

    raw_df = pd.DataFrame(records)

    raw_df.to_csv(
        OUTPUT_RAW,
        index=False,
    )

    summary_rows = []

    metrics = [
        "false_service_alerts",
        "unprotected_bins_visited",
        "unprotected_distance_km",
        "unprotected_fuel_litres",
        "protected_bins_visited",
        "protected_distance_km",
        "protected_fuel_litres",
        "fuel_overhead_unprotected_l",
        "fuel_saved_by_poa_l",
    ]

    for attack_rate in ATTACK_RATES:

        subset = raw_df[
            raw_df["attack_rate"] == attack_rate
        ]

        for metric in metrics:

            stats = summarize(
                subset[metric].to_numpy()
            )

            summary_rows.append(
                {
                    "attack_rate": attack_rate,
                    "metric": metric,
                    "n": stats.n,
                    "mean": stats.mean,
                    "median": stats.median,
                    "std": stats.std,
                    "q1": stats.q1,
                    "q3": stats.q3,
                    "iqr": stats.iqr,
                    "minimum": stats.minimum,
                    "maximum": stats.maximum,
                    "ci95_low": stats.ci95_low,
                    "ci95_high": stats.ci95_high,
                }
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    print("=" * 108)
    print(
        "PHASE 05C — AUTHENTICATED COMPROMISED-NODE FDI "
        "— 100 PAIRED REPLICATES"
    )
    print("=" * 108)

    print(
        f"{'Rate':>6} "
        f"{'False alerts':>15} "
        f"{'Unprotected fuel':>24} "
        f"{'95% CI':>22} "
        f"{'PoA fuel':>14} "
        f"{'Saved by PoA':>16}"
    )

    print("-" * 108)

    for attack_rate in ATTACK_RATES:

        subset = raw_df[
            raw_df["attack_rate"] == attack_rate
        ]

        false_stats = summarize(
            subset[
                "false_service_alerts"
            ].to_numpy()
        )

        fuel_stats = summarize(
            subset[
                "unprotected_fuel_litres"
            ].to_numpy()
        )

        protected_stats = summarize(
            subset[
                "protected_fuel_litres"
            ].to_numpy()
        )

        saved_stats = summarize(
            subset[
                "fuel_saved_by_poa_l"
            ].to_numpy()
        )

        print(
            f"{attack_rate:6.0%} "
            f"{false_stats.mean:15.2f} "
            f"{fuel_stats.mean:24.3f} "
            f"[{fuel_stats.ci95_low:7.3f}, "
            f"{fuel_stats.ci95_high:7.3f}] "
            f"{protected_stats.mean:14.3f} "
            f"{saved_stats.mean:16.3f}"
        )

    print("-" * 108)

    print(
        f"Nominal HDR fuel: "
        f"{nominal.total_fuel_litres:.3f} L"
    )

    print(
        f"Raw observations: {len(raw_df)}"
    )

    print(
        f"Raw CSV: {OUTPUT_RAW}"
    )

    print(
        f"Summary CSV: {OUTPUT_SUMMARY}"
    )

    print("=" * 108)


if __name__ == "__main__":
    main()