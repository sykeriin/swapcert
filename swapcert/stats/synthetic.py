"""Synthetic task populations with a controllable proxy quality, for validating estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Population:
    y_a: np.ndarray  # logged success of A, in {0,1}
    y_b: np.ndarray  # real success of B, in {0,1}
    s_b: np.ndarray  # simulated success of B, in [0,1]
    novelty: np.ndarray  # per-task novelty score in [0,1]

    @property
    def d(self) -> np.ndarray:
        return self.y_b - self.y_a

    @property
    def true_delta(self) -> float:
        return float(self.d.mean())

    def features(self) -> np.ndarray:
        # S_B - Y_A is the natural predictor of D; y_a and novelty let the ridge fix bias.
        return np.column_stack([self.s_b - self.y_a, self.y_a, self.novelty])


def make_population(
    n_tasks: int = 278,
    p_a: float = 0.5,
    delta: float = -0.1,
    sim_quality: float = 0.7,
    discord: float = 0.3,
    rng: np.random.Generator | None = None,
) -> Population:
    """B flips some of A's outcomes; the simulator sees B's true outcome through noise.

    sim_quality in [0,1]: probability the simulator reports B's true outcome (else a coin flip).
    """
    rng = rng or np.random.default_rng()
    y_a = (rng.random(n_tasks) < p_a).astype(float)
    # B loses a fraction `down` of A's successes and rescues a fraction `up` of A's failures.
    # With p_a=0.5:  delta = 0.5(up-down),  P(D != 0) = 0.5(up+down) = discord.
    # General p_a: solve  p_a*down + (1-p_a)*up = discord,  (1-p_a)*up - p_a*down = delta.
    up = np.clip((discord + delta) / (2 * (1 - p_a)), 0.0, 1.0)
    down = np.clip((discord - delta) / (2 * p_a), 0.0, 1.0)
    p_b = np.where(y_a == 1, 1 - down, up)
    y_b = (rng.random(n_tasks) < p_b).astype(float)
    # Novelty is drawn independently of outcomes; the simulator is less reliable on novel tasks
    # (this is the mechanism the out-of-trace detector is meant to exploit).
    nov = rng.random(n_tasks)
    informed = rng.random(n_tasks) < sim_quality * (1 - 0.6 * nov)
    s_b = np.where(informed, y_b, (rng.random(n_tasks) < 0.5).astype(float))
    return Population(y_a, y_b, s_b, nov)
