"""Ship / no-ship decision and sample-size planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from scipy import stats as sst

from .estimators import Estimate


class Verdict(str, Enum):
    SHIP = "SHIP"
    NO_SHIP = "NO_SHIP"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    delta_hat: float
    lo: float
    hi: float
    tolerance: float
    reason: str


def decide(est: Estimate, tolerance: float) -> Decision:
    """Non-inferiority test: tolerance δ is the largest success-rate drop we accept.

    SHIP         if lower bound > -δ   (drop is certified to be smaller than δ)
    NO_SHIP      if upper bound < -δ   (drop is certified to be larger than δ)
    INCONCLUSIVE otherwise -- more real runs are needed.
    """
    if est.lo > -tolerance:
        v, why = Verdict.SHIP, f"lower bound {est.lo:+.3f} > -δ ({-tolerance:+.3f})"
    elif est.hi < -tolerance:
        v, why = Verdict.NO_SHIP, f"upper bound {est.hi:+.3f} < -δ ({-tolerance:+.3f})"
    else:
        v, why = Verdict.INCONCLUSIVE, f"interval [{est.lo:+.3f}, {est.hi:+.3f}] straddles -δ ({-tolerance:+.3f})"
    return Decision(v, est.delta, est.lo, est.hi, tolerance, why)


def true_verdict(true_delta: float, tolerance: float) -> Verdict:
    """Ground-truth call from the exhaustive-run Δ."""
    return Verdict.SHIP if true_delta > -tolerance else Verdict.NO_SHIP


def runs_needed(resid_sd: float, half_width: float, alpha: float = 0.05, n_total: int | None = None) -> int:
    """Real runs for a CLT interval of the given half-width, given residual SD.

    resid_sd is the SD of D - λf on the labelled tasks (use the plain SD of D for real-only).
    Applies a finite-population correction when n_total is given.
    """
    z = float(sst.norm.ppf(1 - alpha / 2))
    n0 = (z * resid_sd / half_width) ** 2
    if n_total:
        n0 = n0 / (1 + n0 / n_total)
    return int(n0) + 1
