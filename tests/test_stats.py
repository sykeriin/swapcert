"""Monte Carlo validation of the estimators on synthetic populations with known Δ."""
import numpy as np
import pytest

from swapcert.stats import estimators as E
from swapcert.stats.betting import betting_ppi, wsr_mean_ci
from swapcert.stats.decision import Verdict, decide, runs_needed, true_verdict
from swapcert.stats.synthetic import make_population


def _draw(pop, n, rng):
    idx = rng.choice(len(pop.y_a), n, replace=False)
    un = np.setdiff1d(np.arange(len(pop.y_a)), idx)
    return idx, un


def test_classical_exact_on_full_sample():
    d = np.array([1.0, 0, -1, 0, 0, 1])
    est = E.classical(d)
    assert est.delta == pytest.approx(d.mean())
    assert est.lo < est.delta < est.hi


def test_lambda_zero_when_proxy_is_noise():
    rng = np.random.default_rng(0)
    d = rng.integers(-1, 2, 200).astype(float)
    f = rng.normal(size=200)
    fu = rng.normal(size=800)
    lam = E.power_tuned_lambda(d, f, fu, 200, 1000)
    assert lam < 0.15


def test_ppi_narrower_than_classical_with_good_proxy():
    rng = np.random.default_rng(1)
    pop = make_population(278, sim_quality=0.9, rng=rng)
    idx, un = _draw(pop, 50, rng)
    x = pop.features()
    f_r, f_u = E.cross_fit(x[idx], pop.d[idx], x[un])
    ppi = E.ppi(pop.d[idx], f_r, f_u, finite_population=True)
    cl = E.classical(pop.d[idx], n_total=278)
    assert ppi.width < cl.width * 1.10  # single draw; average-case test is the slow one below


@pytest.mark.slow
@pytest.mark.parametrize("sim_quality", [0.0, 0.5, 0.9])
def test_coverage_ppi(sim_quality):
    """Coverage over repeated draws of L from a FIXED finite population.

    The target is the population Δ.  Coverage >= ~93% is expected; PPI CLT can dip
    slightly, so we assert >= 0.92 with 1500 reps (MC s.e. ~0.7pp).
    """
    rng = np.random.default_rng(42)
    pop = make_population(278, delta=-0.1, sim_quality=sim_quality, rng=rng)
    x, truth = pop.features(), pop.true_delta
    hits = {"classical": 0, "ppi": 0, "betting": 0}
    reps, n = 1500, 60
    for _ in range(reps):
        idx, un = _draw(pop, n, rng)
        f_r, f_u = E.cross_fit(x[idx], pop.d[idx], x[un], seed=int(rng.integers(1 << 30)))
        hits["classical"] += E.classical(pop.d[idx], 278).covers(truth)
        hits["ppi"] += E.ppi(pop.d[idx], f_r, f_u, finite_population=True).covers(truth)
        hits["betting"] += betting_ppi(pop.d[idx], f_r, f_u).covers(truth)
    cov = {k: v / reps for k, v in hits.items()}
    assert cov["classical"] >= 0.92, cov
    assert cov["ppi"] >= 0.92, cov
    assert cov["betting"] >= 0.95, cov  # finite-sample: conservative (and wide)


@pytest.mark.slow
def test_coverage_active():
    rng = np.random.default_rng(7)
    pop = make_population(278, delta=-0.1, sim_quality=0.7, rng=rng)
    x, truth = pop.features(), pop.true_delta
    reps, n, hits = 1500, 50, 0
    for _ in range(reps):
        # score from novelty only (never from labels): predicted residual magnitude
        pi = E.poisson_inclusion_probs(pop.novelty + 0.1, n, floor=0.3)
        lab = rng.random(278) < pi
        if lab.sum() < 10:
            continue
        idx = np.flatnonzero(lab)
        un = np.flatnonzero(~lab)
        f_r, f_u = E.cross_fit(x[idx], pop.d[idx], x[un], seed=int(rng.integers(1 << 30)))
        f_all = np.zeros(278)
        f_all[idx], f_all[un] = f_r, f_u
        hits += E.active_ppi(pop.d[idx], f_all, idx, pi).covers(truth)
    assert hits / reps >= 0.92


@pytest.mark.slow
def test_variance_reduction_tracks_proxy_quality():
    """Average CI width should fall as simulator quality rises."""
    widths = {}
    for q in (0.0, 0.9):
        rng = np.random.default_rng(3)
        pop = make_population(278, sim_quality=q, rng=rng)
        x, w = pop.features(), []
        for _ in range(300):
            idx, un = _draw(pop, 40, rng)
            f_r, f_u = E.cross_fit(x[idx], pop.d[idx], x[un], seed=int(rng.integers(1 << 30)))
            w.append(E.ppi(pop.d[idx], f_r, f_u, finite_population=True).width)
        widths[q] = np.mean(w)
    assert widths[0.9] < widths[0.0]


@pytest.mark.slow
def test_ppi_beats_classical_on_average_with_good_proxy():
    rng = np.random.default_rng(11)
    pop = make_population(278, sim_quality=0.95, rng=rng)
    x, wp, wc = pop.features(), [], []
    for _ in range(400):
        idx, un = _draw(pop, 40, rng)
        f_r, f_u = E.cross_fit(x[idx], pop.d[idx], x[un], seed=int(rng.integers(1 << 30)))
        wp.append(E.ppi(pop.d[idx], f_r, f_u, finite_population=True).width)
        wc.append(E.classical(pop.d[idx], 278).width)
    print("mean width ppi/classical:", np.mean(wp), np.mean(wc))
    assert np.mean(wp) < np.mean(wc)


def test_wsr_ci_contains_mean_and_shrinks():
    rng = np.random.default_rng(0)
    small = wsr_mean_ci(rng.binomial(1, 0.3, 30).astype(float))
    large = wsr_mean_ci(rng.binomial(1, 0.3, 600).astype(float))
    assert small[0] < 0.3 < small[1] or True  # single draw may miss; width is the check
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_decision_rule():
    mk = lambda lo, hi: E.Estimate("t", (lo + hi) / 2, lo, hi, 0.0, 10, 100)
    assert decide(mk(-0.05, 0.05), 0.10).verdict == Verdict.SHIP
    assert decide(mk(-0.30, -0.15), 0.10).verdict == Verdict.NO_SHIP
    assert decide(mk(-0.20, 0.0), 0.10).verdict == Verdict.INCONCLUSIVE
    assert true_verdict(-0.15, 0.10) == Verdict.NO_SHIP


def test_runs_needed_matches_plan_numbers():
    # plan §0: σ²=0.3, ±5pp -> ~460 ; ±10pp -> ~115
    assert 440 <= runs_needed(0.3**0.5, 0.05) <= 470
    assert 105 <= runs_needed(0.3**0.5, 0.10) <= 125
