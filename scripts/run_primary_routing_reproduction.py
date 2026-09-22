from __future__ import annotations

from pathlib import Path

import pandas as pd

from smart_waste.attacks.fdi import generate_fdi_attack
from smart_waste.core.city import City
from smart_waste.routing.hdr_global import (
    simulate_global_greedy_hdr,
)
from smart_waste.routing.tsr import simulate_tsr
from smart_waste.statistics.descriptive import summarize


MASTER_SEED = 20260922

ATTACK_RATES = [
    0.00,
    0.10,
    0.15,
    0.20,
    0.30,
]

REPETITIONS_PER_SCENARIO = 100

THRESHOLD_PERCENT = 80.0

OUTPUT_RAW = Path(
    "results/raw/"
    "primary_routing_reproduction.csv"
)

OUTPUT_SUMMARY = Path(
    "results/processed/"
    "primary_routing_reproduction_summary.csv"
)


def run_hdr(city, reported=None):
    return simulate_global_greedy_hdr(
        city,
        num_trucks=10,
        truck_capacity_tonnes=10.0,
        fuel_efficiency_km_per_litre=2.5,
        bin_volume_m3=1.1,
        waste_density_kg_per_m3=229.088,
        threshold_percent=THRESHOLD_PERCENT,
        reported_fill_percent=reported,
    )


def main():

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

    # --------------------------------------------------
    # PRIMARY NOMINAL BASELINES
    # --------------------------------------------------

    tsr = simulate_tsr(
        city,
        num_trucks=10,
        truck_capacity_tonnes=10.0,
        fuel_efficiency_km_per_litre=2.5,
        bin_volume_m3=1.1,
        waste_density_kg_per_m3=229.088,
    )

    hdr_nominal = run_hdr(city)

    # --------------------------------------------------
    # PRIMARY FDI SCENARIOS
    # --------------------------------------------------

    records = []

    for attack_rate in ATTACK_RATES:

        for replicate_id in range(
            REPETITIONS_PER_SCENARIO
        ):

            scenario = generate_fdi_attack(
                city,
                master_seed=MASTER_SEED,
                attack_rate=attack_rate,
                attack_type=(
                    "external_unauthenticated"
                ),
                forged_fill_percent=100.0,
                threshold_percent=(
                    THRESHOLD_PERCENT
                ),
                replicate_id=replicate_id,
                selection_mode=(
                    "independent_by_rate"
                ),
            )

            # Unprotected HDR sees forged telemetry.
            unprotected = run_hdr(
                city,
                scenario.reported_fill_overrides(),
            )

            # PoA-filtered routing receives only
            # authenticated telemetry.
            protected = run_hdr(
                city,
                scenario.poa_accepted_overrides(),
            )

            records.append(
                {
                    "master_seed": MASTER_SEED,
                    "attack_rate": attack_rate,
                    "replicate_id": replicate_id,
                    "selection_seed": (
                        scenario.selection_seed
                    ),
                    "attacked_nodes": (
                        scenario.num_attacked_nodes
                    ),
                    "false_service_alerts": len(
                        scenario.false_service_alert_ids(
                            THRESHOLD_PERCENT
                        )
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
                    "poa_bins_visited": (
                        protected.total_bins_visited
                    ),
                    "poa_distance_km": (
                        protected.total_distance_km
                    ),
                    "poa_fuel_litres": (
                        protected.total_fuel_litres
                    ),
                }
            )

    raw_df = pd.DataFrame(records)

    raw_df.to_csv(
        OUTPUT_RAW,
        index=False,
    )

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    summary_rows = []

    for attack_rate in ATTACK_RATES:

        subset = raw_df[
            raw_df["attack_rate"]
            == attack_rate
        ]

        for metric in [
            "false_service_alerts",
            "unprotected_bins_visited",
            "unprotected_distance_km",
            "unprotected_fuel_litres",
            "poa_bins_visited",
            "poa_distance_km",
            "poa_fuel_litres",
        ]:

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

    # --------------------------------------------------
    # REPORT
    # --------------------------------------------------

    print("=" * 112)
    print(
        "PRIMARY ROUTING REPRODUCTION "
        "— MANUSCRIPT METHODOLOGY"
    )
    print("=" * 112)

    print()
    print("NOMINAL BASELINES")
    print("-" * 112)

    print(
        f"TSR bins       : "
        f"{tsr.total_bins_visited}"
    )

    print(
        f"TSR distance   : "
        f"{tsr.total_distance_km:.3f} km"
    )

    print(
        f"TSR fuel       : "
        f"{tsr.total_fuel_litres:.3f} L"
    )

    print()

    print(
        f"HDR bins       : "
        f"{hdr_nominal.total_bins_visited}"
    )

    print(
        f"HDR distance   : "
        f"{hdr_nominal.total_distance_km:.3f} km"
    )

    print(
        f"HDR fuel       : "
        f"{hdr_nominal.total_fuel_litres:.3f} L"
    )

    print()
    print("FDI SCENARIOS")
    print("-" * 112)

    print(
        f"{'Rate':>6} "
        f"{'Attacked':>10} "
        f"{'False alerts':>15} "
        f"{'Unprotected fuel':>21} "
        f"{'95% CI':>22} "
        f"{'PoA fuel':>14}"
    )

    print("-" * 112)

    for rate in ATTACK_RATES:

        subset = raw_df[
            raw_df["attack_rate"] == rate
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

        poa_stats = summarize(
            subset[
                "poa_fuel_litres"
            ].to_numpy()
        )

        attacked = int(
            subset["attacked_nodes"].iloc[0]
        )

        print(
            f"{rate:6.0%} "
            f"{attacked:10d} "
            f"{false_stats.mean:15.2f} "
            f"{fuel_stats.mean:21.3f} "
            f"[{fuel_stats.ci95_low:7.3f}, "
            f"{fuel_stats.ci95_high:7.3f}] "
            f"{poa_stats.mean:14.3f}"
        )

    print("-" * 112)

    print()
    print("MANUSCRIPT REFERENCE VALUES")
    print("-" * 112)

    print(
        "Reported nominal TSR : 100 L"
    )

    print(
        "Reported nominal HDR : 62 L"
    )

    print(
        "Reported attack HDR  : 120 L"
    )

    print()
    print(
        "These reference values are NOT "
        "used for calibration."
    )

    print()
    print(
        f"Raw rows: {len(raw_df)}"
    )

    print(
        f"Raw file: {OUTPUT_RAW}"
    )

    print(
        f"Summary file: {OUTPUT_SUMMARY}"
    )

    print("=" * 112)


if __name__ == "__main__":
    main()