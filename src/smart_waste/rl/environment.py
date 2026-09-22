from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class RewardWeights:
    collection: float = 1.0
    distance: float = 0.5
    hazard_delay: float = 5.0
    sla_violation: float = 3.0


class SmartWasteRoutingEnv(gym.Env):
    """
    Centralized sequential routing environment.

    Observation:
        L_t:
            normalized x/y coordinates of the active truck

        F_t:
            verified fill level of every bin, normalized to [0, 1]

        H_t:
            hazard severity of every bin, normalized to [0, 1]

        T_t:
            elapsed full-bin time for every bin,
            normalized against the 6-hour SLA limit

    Action:
        Select the next bin for the active truck.

    Invalid/unverified bins are removed through action masking.

    Reconstruction assumptions are explicit:
        - active trucks rotate round-robin
        - prediction horizon is externally supplied
        - decision interval is externally supplied
        - collection reward is normalized collected fill
        - unresolved hazard delay is accumulated over time
    """

    metadata = {
        "render_modes": [],
    }

    def __init__(
        self,
        *,
        bin_xy: np.ndarray,
        depot_xy: np.ndarray,
        initial_fill_percent: np.ndarray,
        fill_rate_percent_per_hour: np.ndarray,
        hazard_severity: np.ndarray,
        prediction_horizon_hours: float,
        decision_interval_hours: float,
        verified_mask: np.ndarray | None = None,
        initial_full_elapsed_hours: np.ndarray | None = None,
        num_trucks: int = 10,
        truck_capacity_tonnes: float = 10.0,
        fuel_efficiency_km_per_litre: float = 2.5,
        full_bin_mass_tonnes: float = 0.2519968,
        near_full_threshold_percent: float = 90.0,
        sla_limit_hours: float = 6.0,
        max_steps: int = 1000,
        reward_weights: RewardWeights | None = None,
    ) -> None:

        super().__init__()

        self.bin_xy = np.asarray(
            bin_xy,
            dtype=np.float64,
        )

        self.depot_xy = np.asarray(
            depot_xy,
            dtype=np.float64,
        )

        self.initial_fill_percent = np.asarray(
            initial_fill_percent,
            dtype=np.float64,
        )

        self.fill_rates = np.asarray(
            fill_rate_percent_per_hour,
            dtype=np.float64,
        )

        self.initial_hazard_severity = np.asarray(
            hazard_severity,
            dtype=np.float64,
        )

        if self.bin_xy.ndim != 2:
            raise ValueError(
                "bin_xy must be a 2-D array"
            )

        if self.bin_xy.shape[1] != 2:
            raise ValueError(
                "bin_xy must have shape (N, 2)"
            )

        self.num_bins = self.bin_xy.shape[0]

        if self.depot_xy.shape != (2,):
            raise ValueError(
                "depot_xy must have shape (2,)"
            )

        expected_shape = (self.num_bins,)

        for name, value in [
            (
                "initial_fill_percent",
                self.initial_fill_percent,
            ),
            (
                "fill_rate_percent_per_hour",
                self.fill_rates,
            ),
            (
                "hazard_severity",
                self.initial_hazard_severity,
            ),
        ]:
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape "
                    f"{expected_shape}"
                )

        if np.any(
            self.initial_fill_percent < 0.0
        ) or np.any(
            self.initial_fill_percent > 100.0
        ):
            raise ValueError(
                "fill values must be between 0 and 100"
            )

        if np.any(self.fill_rates < 0.0):
            raise ValueError(
                "fill rates cannot be negative"
            )

        if np.any(
            self.initial_hazard_severity < 0.0
        ) or np.any(
            self.initial_hazard_severity > 1.0
        ):
            raise ValueError(
                "hazard severity must be normalized "
                "to [0, 1]"
            )

        if verified_mask is None:
            self.verified_mask = np.ones(
                self.num_bins,
                dtype=bool,
            )
        else:
            self.verified_mask = np.asarray(
                verified_mask,
                dtype=bool,
            )

            if self.verified_mask.shape != expected_shape:
                raise ValueError(
                    "verified_mask has invalid shape"
                )

        if initial_full_elapsed_hours is None:
            self.initial_full_elapsed_hours = np.zeros(
                self.num_bins,
                dtype=np.float64,
            )
        else:
            self.initial_full_elapsed_hours = np.asarray(
                initial_full_elapsed_hours,
                dtype=np.float64,
            )

            if (
                self.initial_full_elapsed_hours.shape
                != expected_shape
            ):
                raise ValueError(
                    "initial_full_elapsed_hours "
                    "has invalid shape"
                )

        if prediction_horizon_hours <= 0.0:
            raise ValueError(
                "prediction_horizon_hours must be positive"
            )

        if decision_interval_hours <= 0.0:
            raise ValueError(
                "decision_interval_hours must be positive"
            )

        if num_trucks <= 0:
            raise ValueError(
                "num_trucks must be positive"
            )

        if truck_capacity_tonnes <= 0.0:
            raise ValueError(
                "truck_capacity_tonnes must be positive"
            )

        if fuel_efficiency_km_per_litre <= 0.0:
            raise ValueError(
                "fuel efficiency must be positive"
            )

        if full_bin_mass_tonnes <= 0.0:
            raise ValueError(
                "full_bin_mass_tonnes must be positive"
            )

        if sla_limit_hours <= 0.0:
            raise ValueError(
                "sla_limit_hours must be positive"
            )

        if max_steps <= 0:
            raise ValueError(
                "max_steps must be positive"
            )

        self.prediction_horizon_hours = (
            prediction_horizon_hours
        )

        self.decision_interval_hours = (
            decision_interval_hours
        )

        self.num_trucks = num_trucks

        self.truck_capacity_tonnes = (
            truck_capacity_tonnes
        )

        self.fuel_efficiency_km_per_litre = (
            fuel_efficiency_km_per_litre
        )

        self.full_bin_mass_tonnes = (
            full_bin_mass_tonnes
        )

        self.near_full_threshold_percent = (
            near_full_threshold_percent
        )

        self.sla_limit_hours = sla_limit_hours
        self.max_steps = max_steps

        self.reward_weights = (
            reward_weights
            if reward_weights is not None
            else RewardWeights()
        )

        all_xy = np.vstack(
            [
                self.bin_xy,
                self.depot_xy.reshape(1, 2),
            ]
        )

        self.coordinate_min = all_xy.min(
            axis=0
        )

        self.coordinate_max = all_xy.max(
            axis=0
        )

        self.coordinate_span = (
            self.coordinate_max
            - self.coordinate_min
        )

        self.coordinate_span[
            self.coordinate_span == 0.0
        ] = 1.0

        observation_size = (
            2
            + self.num_bins
            + self.num_bins
            + self.num_bins
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_size,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(
            self.num_bins
        )

        self.fill_percent: np.ndarray
        self.hazard_severity: np.ndarray
        self.full_elapsed_hours: np.ndarray

        self.truck_xy: np.ndarray
        self.truck_load_tonnes: np.ndarray

        self.active_truck: int
        self.step_count: int

        self.total_distance_km: float
        self.total_fuel_litres: float

    def _distance(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        return float(
            np.linalg.norm(a - b)
        )

    def _normalize_xy(
        self,
        xy: np.ndarray,
    ) -> np.ndarray:
        return (
            (xy - self.coordinate_min)
            / self.coordinate_span
        )

    def _predicted_fill(self) -> np.ndarray:
        return np.minimum(
            100.0,
            self.fill_percent
            + (
                self.fill_rates
                * self.prediction_horizon_hours
            ),
        )

    def action_mask(self) -> np.ndarray:
        """
        Eligible bins are:
          - verified,
          - already near-full, OR
          - predicted to overflow, OR
          - carrying a hazard.

        Emergency hazard severity == 1.0 overrides
        ordinary distance/fullness candidates.
        """

        near_full = (
            self.fill_percent
            >= self.near_full_threshold_percent
        )

        predicted_overflow = (
            self._predicted_fill()
            >= 100.0
        )

        hazard_candidate = (
            self.hazard_severity > 0.0
        )

        candidates = (
            self.verified_mask
            & (
                near_full
                | predicted_overflow
                | hazard_candidate
            )
        )

        emergency = (
            candidates
            & np.isclose(
                self.hazard_severity,
                1.0,
            )
        )

        if np.any(emergency):
            return emergency

        return candidates

    def _observation(self) -> np.ndarray:
        truck_location = self._normalize_xy(
            self.truck_xy[
                self.active_truck
            ]
        )

        fill_state = (
            self.fill_percent / 100.0
        )

        hazard_state = np.clip(
            self.hazard_severity,
            0.0,
            1.0,
        )

        time_state = np.minimum(
            1.0,
            (
                self.full_elapsed_hours
                / self.sla_limit_hours
            ),
        )

        observation = np.concatenate(
            [
                truck_location,
                fill_state,
                hazard_state,
                time_state,
            ]
        )

        return observation.astype(
            np.float32
        )

    def _info(self) -> dict:
        mask = self.action_mask()

        return {
            "action_mask": mask.copy(),
            "candidate_count": int(
                np.count_nonzero(mask)
            ),
            "active_truck": self.active_truck,
            "total_distance_km": (
                self.total_distance_km
            ),
            "total_fuel_litres": (
                self.total_fuel_litres
            ),
            "simulated_time_hours": (
                self.simulated_time_hours
            ),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(
            seed=seed
        )

        self.fill_percent = (
            self.initial_fill_percent.copy()
        )

        self.hazard_severity = (
            self.initial_hazard_severity.copy()
        )

        self.full_elapsed_hours = (
            self.initial_full_elapsed_hours.copy()
        )

        self.truck_xy = np.repeat(
            self.depot_xy.reshape(1, 2),
            self.num_trucks,
            axis=0,
        )

        self.truck_load_tonnes = np.zeros(
            self.num_trucks,
            dtype=np.float64,
        )

        self.active_truck = 0
        self.step_count = 0

        self.total_distance_km = 0.0
        self.total_fuel_litres = 0.0
        self.simulated_time_hours = 0.0
        return (
            self._observation(),
            self._info(),
        )

    def step(
        self,
        action: int,
    ):
        if not self.action_space.contains(
            action
        ):
            raise ValueError(
                f"invalid action: {action}"
            )

        mask = self.action_mask()

        if not mask[action]:
            raise ValueError(
                f"action {action} is not eligible "
                "under the current action mask"
            )

        truck_id = self.active_truck

        # The configured decision interval represents one complete
        # fleet scheduling epoch. The centralized sequential
        # coordinator assigns one target to each active truck
        # within that epoch.
        action_elapsed_hours = (
            self.decision_interval_hours
            / self.num_trucks
        )        

        current_location = (
            self.truck_xy[
                truck_id
            ].copy()
        )

        current_fill = float(
            self.fill_percent[action]
        )

        collected_fraction = (
            current_fill / 100.0
        )

        demand_tonnes = (
            self.full_bin_mass_tonnes
            * collected_fraction
        )

        dispatch_distance = 0.0
        terminal_return_distance = 0.0

        capacity_returned_before_service = False

        if (
            self.truck_load_tonnes[truck_id]
            + demand_tonnes
            > self.truck_capacity_tonnes
        ):
            dispatch_distance += self._distance(
                current_location,
                self.depot_xy,
            )

            current_location = (
                self.depot_xy.copy()
            )

            self.truck_load_tonnes[
                truck_id
            ] = 0.0

            capacity_returned_before_service = True

        dispatch_distance += self._distance(
            current_location,
            self.bin_xy[action],
        )

        self.truck_xy[
            truck_id
        ] = self.bin_xy[action]

        self.truck_load_tonnes[
            truck_id
        ] += demand_tonnes

        # Service the selected bin.
        self.fill_percent[action] = 0.0
        self.hazard_severity[action] = 0.0
        self.full_elapsed_hours[action] = 0.0

        # Advance physical time.
        self.fill_percent = np.minimum(
            100.0,
            self.fill_percent
            + (
                self.fill_rates
                * action_elapsed_hours
            ),
        )

        full_now = (
            self.fill_percent >= 100.0
        )

        self.full_elapsed_hours = np.where(
            full_now,
            (
                self.full_elapsed_hours
                + action_elapsed_hours
            ),
            0.0,
        )

        # Hazard delay represents unresolved verified
        # hazards remaining after the current service.
        hazard_delay_term = float(
            np.sum(
                self.hazard_severity[
                    self.verified_mask
                ]
            )
            * action_elapsed_hours
        )

        sla_violation_term = (
            1.0
            if np.any(
                self.verified_mask
                & (
                    self.full_elapsed_hours
                    > self.sla_limit_hours
                )
            )
            else 0.0
        )

        self.simulated_time_hours += (
            action_elapsed_hours
        )

        self.step_count += 1

        self.active_truck = (
            self.active_truck + 1
        ) % self.num_trucks

        next_mask = self.action_mask()

        terminated = not np.any(
            next_mask
        )

        truncated = (
            self.step_count
            >= self.max_steps
        )

        if terminated or truncated:
            for truck_index in range(
                self.num_trucks
            ):
                terminal_return_distance += (
                    self._distance(
                        self.truck_xy[
                            truck_index
                        ],
                        self.depot_xy,
                    )
                )

                self.truck_xy[
                    truck_index
                ] = self.depot_xy

        step_distance = (
            dispatch_distance
            + terminal_return_distance
        )

        step_fuel = (
            step_distance
            / self.fuel_efficiency_km_per_litre
        )

        self.total_distance_km += (
            step_distance
        )

        self.total_fuel_litres += (
            step_fuel
        )

        reward = (
            self.reward_weights.collection
            * collected_fraction
            - self.reward_weights.distance
            * step_distance
            - self.reward_weights.hazard_delay
            * hazard_delay_term
            - self.reward_weights.sla_violation
            * sla_violation_term
        )

        observation = self._observation()

        info = self._info()

        info.update(
            {
                "selected_bin": int(action),
                "collected_fraction": (
                    collected_fraction
                ),
                "collected_tonnes": (
                    demand_tonnes
                ),
                "dispatch_distance_km": (
                    dispatch_distance
                ),
                "terminal_return_distance_km": (
                    terminal_return_distance
                ),
                "step_distance_km": (
                    step_distance
                ),
                "step_fuel_litres": (
                    step_fuel
                ),
                "hazard_delay_term": (
                    hazard_delay_term
                ),
                "sla_violation_term": (
                    sla_violation_term
                ),
                "capacity_returned_before_service": (
                    capacity_returned_before_service
                ),
            }
        )

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )