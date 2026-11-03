"""Offline resampling harness.

Given per-task arrays for ONE swap pair (all computed once and cached):
  y_a, y_b   real outcomes of A (logs) and B (exhaustive live runs = ground truth)
  s_b        simulated outcome of B (forked simulation)
  nov        novelty score
repeatedly draw the "real budget" subset L, run each method as if only L had real labels, and
tabulate coverage, width, and decision accuracy against the exhaustive-run truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swapcert.stats import estimators as E
from swapcert.stats.betting import betting_classical
from swapcert.stats.decision import Verdict, decide, true_verdict


@dataclass
class Row:
    method: str
    n: int
    coverage: float
    width: float
    acc_decisive: float  # correct calls among decisive calls
    p_inconclusive: float
    p_correct: float  # P(decisive AND correct) -- the headline quantity


def feature_sets(y_a, s_b, nov, replay=None) -> dict[str, np.ndarray | None]:
    d_sim = (s_b - y_a)[:, None]
    fs: dict[str, np.ndarray | None] = {
        "P0 real-only": None,
        "P1 A-as-proxy": y_a[:, None],  # predict D from A's logged outcome only (no simulation)
        "P5 sim": d_sim,
        "P6 sim+novelty": np.column_stack([d_sim, nov, y_a]),
    }
    if replay is not None:
        fs["P2 replay"] = (replay - y_a)[:, None]
    return fs


def run(y_a, y_b, s_b, nov, budgets, tolerance=0.10, reps=500, seed=0, alpha=0.05, replay=None,
        active=False, active_floor=0.3) -> list[Row]:
    rng = np.random.default_rng(seed)
    d = y_b - y_a
    big_n = len(d)
    truth = float(d.mean())
    tv = true_verdict(truth, tolerance)
    fsets = feature_sets(y_a, s_b, nov, replay)
    rows: list[Row] = []
    for n in budgets:
        acc: dict[str, list] = {k: [] for k in list(fsets) + ["betting-real"] + (["P6 active"] if active else [])}
        for _ in range(reps):
            idx = rng.choice(big_n, n, replace=False)
            un = np.setdiff1d(np.arange(big_n), idx)
            for name, x in fsets.items():
                if x is None:
                    est = E.classical(d[idx], big_n, alpha)
                else:
                    f_r, f_u = E.cross_fit(x[idx], d[idx], x[un], seed=int(rng.integers(1 << 30)))
                    est = E.ppi(d[idx], f_r, f_u, alpha, finite_population=True)
                acc[name].append((est, decide(est, tolerance).verdict))
            est = betting_classical(d[idx], alpha)
            acc["betting-real"].append((est, decide(est, tolerance).verdict))
            if active:  # novelty-weighted Poisson sampling with IPW (known inclusion probabilities)
                pi = E.poisson_inclusion_probs(nov + 0.1, n, floor=active_floor)
                lab = np.flatnonzero(rng.random(big_n) < pi)
                if len(lab) >= 10:
                    ul = np.setdiff1d(np.arange(big_n), lab)
                    x = fsets["P6 sim+novelty"]
                    f_r, f_u = E.cross_fit(x[lab], d[lab], x[ul], seed=int(rng.integers(1 << 30)))
                    f_all = np.zeros(big_n)
                    f_all[lab], f_all[ul] = f_r, f_u
                    est = E.active_ppi(d[lab], f_all, lab, pi, alpha)
                    acc["P6 active"].append((est, decide(est, tolerance).verdict))
        for name, res in acc.items():
            if not res:
                continue
            cov = np.mean([e.covers(truth) for e, _ in res])
            w = np.mean([e.width for e, _ in res])
            decisive = [v for _, v in res if v != Verdict.INCONCLUSIVE]
            correct = sum(v == tv for v in decisive)
            rows.append(Row(name, n, float(cov), float(w),
                            correct / len(decisive) if decisive else float("nan"),
                            1 - len(decisive) / len(res), correct / len(res)))
    return rows


def runs_to_reach(rows: list[Row], method: str, p: float = 0.8) -> int | None:
    """Smallest budget at which P(decisive & correct) >= p."""
    for r in sorted((r for r in rows if r.method == method), key=lambda r: r.n):
        if r.p_correct >= p:
            return r.n
    return None
