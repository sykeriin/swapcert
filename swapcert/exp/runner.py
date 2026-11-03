"""Multi-pair toy experiment: several swap pairs -> coverage/width/decision tables + figures.

TOY ONLY.  The agents are scripted mocks and the simulator is a mock, so these numbers validate the
plumbing and the statistics, not any claim about real LLM agents.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from swapcert.envs.toy import MockAgent
from swapcert.exp import resample as R
from swapcert.exp.figures import line_chart_svg
from swapcert.exp.toy_experiment import build_dataset
from swapcert.stats import estimators as E
from swapcert.stats.decision import decide, true_verdict

A = MockAgent("A", skill=0.5)
PAIRS = {  # name -> candidate B
    "A->A control": MockAgent("A", skill=0.5),
    "style change": MockAgent("B1", skill=0.5, style=1),
    "mild downgrade": MockAgent("B2", skill=0.4, style=1),
    "large downgrade": MockAgent("B3", skill=0.1, style=1),
}
MARGIN = 0.02  # pairs whose true delta lies within MARGIN of -tolerance are 'borderline' and excluded from accuracy


def run_experiment(out_dir: str | Path, n_tasks: int = 200, tolerance: float = 0.05, budgets=(30, 60, 100),
                   reps: int = 150, sim_error: float = 0.1, seed: int = 0, active: bool = False) -> dict:
    budgets = tuple(b for b in budgets if b < n_tasks)  # a budget covering every task leaves nothing to infer
    if not budgets:
        raise ValueError(f"all budgets >= n_tasks={n_tasks}; use fewer real runs than tasks")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    results: dict = {"tolerance": tolerance, "n_tasks": n_tasks, "pairs": {}}
    pooled: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    width: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for name, agent_b in PAIRS.items():
        D = build_dataset(n_tasks, A, agent_b, sim_error=sim_error, seed=seed)
        d = D["y_b"] - D["y_a"]
        truth = float(d.mean())
        borderline = abs(truth + tolerance) < MARGIN
        rows = R.run(D["y_a"], D["y_b"], D["s_b"], D["nov"], list(budgets), tolerance, reps, seed, active=active)
        # one concrete decision report from a single uniform draw of the middle budget
        n_mid = budgets[len(budgets) // 2]
        idx = rng.choice(n_tasks, n_mid, replace=False)
        un = np.setdiff1d(np.arange(n_tasks), idx)
        x = np.column_stack([D["s_b"] - D["y_a"], D["nov"], D["y_a"]])
        f_r, f_u = E.cross_fit(x[idx], d[idx], x[un], seed=seed)
        est = E.ppi(d[idx], f_r, f_u, finite_population=True)
        dec = decide(est, tolerance)
        results["pairs"][name] = {
            "true_delta": truth, "true_verdict": true_verdict(truth, tolerance).value, "borderline": borderline,
            "single_draw": {"n_real": n_mid, "delta_hat": est.delta, "ci": [est.lo, est.hi],
                            "verdict": dec.verdict.value, "covers_truth": bool(est.covers(truth))},
            "never_diverged_frac": float((D["diverged_at"] == -1).mean()),
            "corr_sim_vs_delta": float(np.corrcoef(D["s_b"] - D["y_a"], d)[0, 1]) if d.std() > 0 else None,
            "rows": [r.__dict__ for r in rows],
        }
        if not borderline:
            for r in rows:
                pooled[r.method][r.n].append(r.p_correct)
        for r in rows:
            width[r.method][r.n].append(r.width)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    series = {m: [(n, float(np.mean(v))) for n, v in sorted(by_n.items())] for m, by_n in pooled.items()}
    line_chart_svg(series, "P(correct decisive call) vs real runs (toy, mean over non-borderline pairs)",
                   "real runs of B", "P(correct)", out / "fig_correct.svg", ylim=(0, 1))
    wser = {m: [(n, float(np.mean(v))) for n, v in sorted(by_n.items())] for m, by_n in width.items()}
    line_chart_svg(wser, "95% CI width vs real runs (toy, mean over all pairs)", "real runs of B",
                   "CI width", out / "fig_width.svg")
    (out / "summary.md").write_text(_summary(results))
    return results


def _summary(res: dict) -> str:
    n_mid = next(iter(res["pairs"].values()))["single_draw"]["n_real"]
    L = [f"# Toy experiment summary (tolerance {res['tolerance']}, {res['n_tasks']} tasks)", "",
         "> Toy agents and a mock simulator. Validates plumbing and statistics only; no claim about real LLM agents.",
         ("> The A->A control is not exactly 0: A's logs are one recorded run per task, the control's truth "
          "averages fresh runs, so sampling noise remains by construction."), "",
         (f"| Pair | true Δ | true call | borderline | single-draw call (n={n_mid}) | CI | CI covers truth | "
          "B never diverged | corr(sim, Δ) |"), "|---|---|---|---|---|---|---|---|---|"]
    for name, p in res["pairs"].items():
        s = p["single_draw"]
        cor = "n/a" if p["corr_sim_vs_delta"] is None else f"{p['corr_sim_vs_delta']:.2f}"
        L.append(f"| {name} | {p['true_delta']:+.3f} | {p['true_verdict']} | {p['borderline']} | {s['verdict']} | "
                 f"[{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] | {s['covers_truth']} | {p['never_diverged_frac']:.0%} | {cor} |")
    L += ["", "Method rows (coverage / width / P(correct)) per pair are in `results.json`.",
          "Figures: `fig_correct.svg`, `fig_width.svg`."]
    return "\n".join(L) + "\n"
