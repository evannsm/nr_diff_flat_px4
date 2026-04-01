from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NRDiffFlatProfileConfig:
    name: str
    lookahead_horizon_s: float
    alpha: np.ndarray
    integral_gain: np.ndarray
    integral_limit: np.ndarray
    num_iterations: int
    iteration_damping: float
    use_thrust_cbf: bool


def build_nr_profile(profile_name: str) -> NRDiffFlatProfileConfig:
    """Return a documented diff-flat profile for repeatable experiments."""
    profile = profile_name.strip().lower()
    if profile == "baseline":
        return NRDiffFlatProfileConfig(
            name="baseline",
            lookahead_horizon_s=0.8,
            alpha=np.array([20.0, 30.0, 30.0, 30.0], dtype=np.float64),
            integral_gain=np.zeros(4, dtype=np.float64),
            integral_limit=np.zeros(4, dtype=np.float64),
            num_iterations=1,
            iteration_damping=1.0,
            use_thrust_cbf=True,
        )
    if profile == "workshop":
        return NRDiffFlatProfileConfig(
            name="workshop",
            lookahead_horizon_s=0.5,
            alpha=np.array([24.0, 34.0, 34.0, 24.0], dtype=np.float64),
            integral_gain=np.array([0.30, 0.30, 0.45, 0.10], dtype=np.float64),
            integral_limit=np.array([0.75, 0.75, 0.50, 0.30], dtype=np.float64),
            num_iterations=2,
            iteration_damping=0.65,
            use_thrust_cbf=True,
        )
    raise ValueError(f"Unknown NR diff-flat profile: {profile_name}")
