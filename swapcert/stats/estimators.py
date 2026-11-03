"""Estimators for the paired change in success rate, Δ = mean_t (Y_B(t) - Y_A(t)).

Everything works on the per-task difference D_t = Y_B(t) - Y_A(t).  D_t is observed only for
the labelled ("real") tasks L.  A proxy f_t (e.g. simulated success of B minus logged success
of A) is available for all N tasks.

Validity does not depend on the proxy: the correction term uses real labels, so a bad proxy
only widens the interval.  That holds when L is sampled uniformly (or with known inclusion
probabilities) -- see `active_ppi`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sst


@dataclass(frozen=True)
class Estimate:
    method: str
    delta: float
    lo: float
    hi: float
    se: float
    n_real: int
    n_total: int
    lam: float | None = None  # PPI++ power-tuning weight actually used

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def covers(self, truth: float) -> bool:
        return self.lo <= truth <= self.hi


def _z(alpha: float) -> float:
    return float(sst.norm.ppf(1 - alpha / 2))


def classical(d_real: np.ndarray, n_total: int | None = None, alpha: float = 0.05) -> Estimate:
    """Real-only mean of D with a t-interval (baseline P0).

    If `n_total` is given, a finite-population correction is applied (L drawn without
    replacement from n_total tasks).
    """
    d = np.asarray(d_real, float)
    n = len(d)
    if n < 2:
        raise ValueError("need at least 2 real runs")
    var = d.var(ddof=1)
    fpc = 1.0 if n_total is None else max(1.0 - n / n_total, 0.0)
    se = float(np.sqrt(var / n * fpc))
    t = float(sst.t.ppf(1 - alpha / 2, n - 1))
    m = float(d.mean())
    return Estimate("classical", m, m - t * se, m + t * se, se, n, n_total or n)


def power_tuned_lambda(
    d: np.ndarray, f: np.ndarray, f_all: np.ndarray, n: int, big_n: int, finite: bool = False
) -> float:
    """PPI++ optimal lambda, clipped to [0, 1].

    superpopulation target: Cov(D,f) / ((1 + n/N) Var(f))
    finite-population target: Cov(D,f) / Var(f)   (the proxy mean over all N is exact)
    Clipping is conservative; lambda -> 0 recovers the real-only estimator.
    """
    if len(d) < 3:
        return 0.0
    vf = np.var(np.concatenate([f, f_all]), ddof=1)
    if vf <= 1e-12:
        return 0.0
    cov = np.cov(d, f, ddof=1)[0, 1]
    denom = vf if finite else (1 + n / big_n) * vf
    return float(np.clip(cov / denom, 0.0, 1.0))


def ppi(
    d_real: np.ndarray,
    f_real: np.ndarray,
    f_unlabeled: np.ndarray,
    alpha: float = 0.05,
    lam: float | None = None,
    finite_population: bool = False,
) -> Estimate:
    """PPI++ mean estimate with CLT interval.

    d_real: D on labelled tasks; f_real: proxy on those same tasks; f_unlabeled: proxy on the
    remaining tasks.  If lam is None it is power-tuned.  The population is N = n + len(unlab).

    Target (choose deliberately):
      finite_population=False  Δ is the mean over the *task distribution* (future traffic).
                               Variance = Var(D-λf)/n + λ²Var(f)/N.
      finite_population=True   Δ is the mean over exactly these N tasks (benchmark ground truth,
                               and any "re-run my whole suite" decision).  L is drawn without
                               replacement, so Var = (1-n/N)·Var(D-λf)/n; no proxy-variance term.

    f must not have been fitted on the labelled tasks it is evaluated on (use `cross_fit`).
    """
    d, f, fu = (np.asarray(a, float) for a in (d_real, f_real, f_unlabeled))
    n, big_n = len(d), len(d) + len(fu)
    f_all = np.concatenate([f, fu])
    if lam is None:
        lam = power_tuned_lambda(d, f, fu, n, big_n, finite_population)
    theta = lam * f_all.mean() + (d - lam * f).mean()
    var_resid = np.var(d - lam * f, ddof=1) / n
    if finite_population:
        se = float(np.sqrt(var_resid * max(1 - n / big_n, 0.0)))
    else:
        var_proxy = lam**2 * np.var(f_all, ddof=1) / big_n
        se = float(np.sqrt(var_resid + var_proxy))
    z = _z(alpha)
    return Estimate("ppi++", float(theta), theta - z * se, theta + z * se, se, n, big_n, lam)


def cross_fit(
    x_real: np.ndarray,
    d_real: np.ndarray,
    x_unlabeled: np.ndarray,
    k: int = 5,
    ridge: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted ridge proxy (Zrnic & Candès 2024).

    Each labelled task is predicted by a model that never saw it; unlabelled tasks are predicted
    by the average of the K fold models.  Returns (f_real, f_unlabeled).
    """
    x_real = np.atleast_2d(np.asarray(x_real, float))
    if x_real.shape[0] != len(d_real):
        x_real = x_real.T
    x_real = x_real.reshape(len(d_real), -1)
    x_un = np.asarray(x_unlabeled, float).reshape(-1, x_real.shape[1])  # tolerates an empty unlabelled set
    n = len(d_real)
    k = min(k, n)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), k)
    f_real = np.zeros(n)
    f_un = np.zeros(len(x_un))
    for idx in folds:
        tr = np.setdiff1d(np.arange(n), idx)
        w = _ridge_fit(x_real[tr], np.asarray(d_real, float)[tr], ridge)
        f_real[idx] = _ridge_predict(w, x_real[idx])
        f_un += _ridge_predict(w, x_un) / k
    return f_real, f_un


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    mu, sd = x.mean(0), x.std(0) + 1e-9
    xs = (x - mu) / sd
    a = np.hstack([np.ones((len(xs), 1)), xs])
    reg = ridge * np.eye(a.shape[1])
    reg[0, 0] = 0.0
    w = np.linalg.solve(a.T @ a + reg, a.T @ y)
    return np.concatenate([[mu.size], mu, sd, w])  # packed: dim, mu, sd, coef


def _ridge_predict(packed: np.ndarray, x: np.ndarray) -> np.ndarray:
    p = int(packed[0])
    mu, sd, w = packed[1 : 1 + p], packed[1 + p : 1 + 2 * p], packed[1 + 2 * p :]
    xs = (x - mu) / sd
    return w[0] + xs @ w[1:]


def poisson_inclusion_probs(
    score: np.ndarray, n_budget: float, floor: float = 0.2
) -> np.ndarray:
    """Inclusion probabilities for active sampling.

    pi_t = (1-floor) * pi*_t + floor * n/N, where pi* is proportional to `score`
    (predicted |residual|) and rescaled to expected size n_budget.  The uniform floor bounds
    the importance weights by N/(floor*n).
    """
    score = np.asarray(score, float)
    big_n = len(score)
    s = np.maximum(score, 1e-9)
    pi_star = np.minimum(n_budget * s / s.sum(), 1.0)
    pi = (1 - floor) * pi_star + floor * n_budget / big_n
    return np.clip(pi, 1e-6, 1.0)


def active_ppi(
    d_real: np.ndarray,
    f_all: np.ndarray,
    labeled_idx: np.ndarray,
    pi: np.ndarray,
    alpha: float = 0.05,
) -> Estimate:
    """Inverse-probability-weighted PPI for non-uniform (Poisson) sampling.

    theta = mean_all f + (1/N) sum_{t in L} (D_t - f_t) / pi_t
    Variance (Poisson sampling): (1/N^2) sum_{t in L} (1-pi_t)/pi_t^2 * r_t^2.
    `f_all` is the proxy for ALL N tasks (cross-fitted for labelled ones).
    """
    f_all = np.asarray(f_all, float)
    idx = np.asarray(labeled_idx)
    pi = np.asarray(pi, float)
    big_n = len(f_all)
    r = np.asarray(d_real, float) - f_all[idx]
    w = 1.0 / pi[idx]
    theta = f_all.mean() + (w * r).sum() / big_n
    se = float(np.sqrt(((1 - pi[idx]) * w**2 * r**2).sum()) / big_n)
    z = _z(alpha)
    return Estimate("active-ppi", float(theta), theta - z * se, theta + z * se, se, len(idx), big_n)
