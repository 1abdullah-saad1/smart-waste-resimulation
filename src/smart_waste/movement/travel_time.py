from __future__ import annotations


def travel_time_hours(
    *,
    distance_km: float,
    speed_km_per_hour: float,
) -> float:
    """Physical road travel time."""

    if distance_km < 0.0:
        raise ValueError(
            "distance_km cannot be negative"
        )

    if speed_km_per_hour <= 0.0:
        raise ValueError(
            "speed_km_per_hour must be positive"
        )

    return distance_km / speed_km_per_hour


def travel_time_minutes(
    *,
    distance_km: float,
    speed_km_per_hour: float,
) -> float:
    return (
        travel_time_hours(
            distance_km=distance_km,
            speed_km_per_hour=speed_km_per_hour,
        )
        * 60.0
    )
