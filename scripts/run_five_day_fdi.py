from pathlib import Path

import pandas as pd

from smart_waste.security.five_day_fdi import (
    simulate_five_day_unauthenticated_fdi,
)


OUTPUT = Path(
    "results/raw/five_day_fdi_primary.csv"
)


def main():

    result = (
        simulate_five_day_unauthenticated_fdi(
            alerts_per_day=10,
            number_of_days=5,
        )
    )

    records = [
        {
            "day": row.day,
            "injected_today": row.injected_today,
            "cumulative_injected": (
                row.cumulative_injected
            ),
            "unprotected_processed_today": (
                row.unprotected_processed_today
            ),
            "unprotected_cumulative_processed": (
                row.unprotected_cumulative_processed
            ),
            "poa_processed_today": (
                row.poa_processed_today
            ),
            "poa_cumulative_processed": (
                row.poa_cumulative_processed
            ),
            "poa_rejected_today": (
                row.poa_rejected_today
            ),
            "poa_cumulative_rejected": (
                row.poa_cumulative_rejected
            ),
        }
        for row in result.daily_results
    ]

    df = pd.DataFrame(records)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 83)
    print(
        "PHASE 06B — FIVE-DAY FDI "
        "PRIMARY MANUSCRIPT REPRODUCTION"
    )
    print("=" * 83)

    print(
        f"{'Day':>5} "
        f"{'Injected':>10} "
        f"{'Cumulative':>12} "
        f"{'Unprotected cloud':>20} "
        f"{'PoA cloud':>12} "
        f"{'PoA rejected':>15}"
    )

    print("-" * 83)

    for row in result.daily_results:

        print(
            f"{row.day:5d} "
            f"{row.injected_today:10d} "
            f"{row.cumulative_injected:12d} "
            f"{row.unprotected_cumulative_processed:20d} "
            f"{row.poa_cumulative_processed:12d} "
            f"{row.poa_cumulative_rejected:15d}"
        )

    print("-" * 83)

    print(
        "Total injected:",
        result.total_injected,
    )

    print(
        "Unprotected reaching cloud:",
        result.total_unprotected_processed,
        f"({result.unprotected_cloud_penetration_rate:.0%})",
    )

    print(
        "PoA reaching cloud:",
        result.total_poa_processed,
        f"({result.poa_cloud_penetration_rate:.0%})",
    )

    print(
        "PoA rejected:",
        result.total_poa_rejected,
        f"({result.poa_rejection_rate:.0%})",
    )

    print(
        "Raw file:",
        OUTPUT,
    )

    print("=" * 83)


if __name__ == "__main__":
    main()