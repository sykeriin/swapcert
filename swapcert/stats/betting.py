"""Finite-sample confidence interval by betting (Waudby-Smith & Ramdas, 2023).

Used as the conservative alternative to CLT intervals when n is small.  For the PPI setting we
bound the residual r = D - f in [-2, 2] (D in [-1, 1], f clipped to [-1, 1]) and shift by the
proxy mean, which is a known constant given a fixed (cross-fitted) proxy.
"""

from __future__ import annotations

import numpy as np

from .estimators import Estimate


def wsr_mean_ci(x: np.ndarray, alpha: float = 0.05, grid: int = 2001) -> tuple[float, float]:
    """Hedged-capital betting CI for the mean of x in [0, 1].

    Returns the hull of the non-rejected means.  Valid at any n (no CLT), for sampling with
    replacement; conservative for sampling without replacement.
    """
    x = np.asarray(x, float)
    n = len(x)
    ms = np.linspace(1e-4, 1 - 1e-4, grid)
    t = np.arange(1, n + 1)
    mu_hat = (0.5 + np.cumsum(x)) / (t + 1)
    s2 = (0.25 + np.cumsum((x - mu_hat) ** 2)) / (t + 1)
    sig_prev = np.concatenate([[0.25], s2[:-1]])
    lam = np.sqrt(2 * np.log(2 / alpha) / (sig_prev * t * np.log(t + 1)))
    lam = lam[:, None]
    lam_p = np.minimum(lam, 0.5 / ms[None, :])
    lam_m = np.minimum(lam, 0.5 / (1 - ms[None, :]))
    diff = x[:, None] - ms[None, :]
    log_kp = np.log1p(lam_p * diff).sum(0)
    log_km = np.log1p(-lam_m * diff).sum(0)
    log_k = np.logaddexp(np.log(0.5) + log_kp, np.log(0.5) + log_km)
    ok = log_k < np.log(1 / alpha)
    if not ok.any():
        return 0.0, 1.0
    return float(ms[ok].min()), float(ms[ok].max())


def betting_ppi(
    d_real: np.ndarray,
    f_real: np.ndarray,
    f_unlabeled: np.ndarray,
    alpha: float = 0.05,
    bound: float = 2.0,
) -> Estimate:
    """Finite-sample PPI interval: mean(f over all) + WSR interval for the mean residual."""
    d = np.asarray(d_real, float)
    f_real = np.clip(np.asarray(f_real, float), -1, 1)
    f_un = np.clip(np.asarray(f_unlabeled, float), -1, 1)
    f_all = np.concatenate([f_real, f_un])
    r = d - f_real
    lo01, hi01 = wsr_mean_ci((r + bound) / (2 * bound), alpha)
    lo_r, hi_r = lo01 * 2 * bound - bound, hi01 * 2 * bound - bound
    base = float(f_all.mean())
    theta = base + float(r.mean())
    se = float("nan")
    return Estimate("betting-ppi", theta, base + lo_r, base + hi_r, se, len(d), len(f_all))


def betting_classical(d_real: np.ndarray, alpha: float = 0.05) -> Estimate:
    """Real-only betting CI for the mean of D in [-1, 1]."""
    d = np.asarray(d_real, float)
    lo01, hi01 = wsr_mean_ci((d + 1) / 2, alpha)
    return Estimate(
        "betting-classical", float(d.mean()), lo01 * 2 - 1, hi01 * 2 - 1, float("nan"), len(d), len(d)
    )
