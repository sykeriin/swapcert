"""Turn (tasks, A's recorded runs, candidate agent B, simulator) into the per-task table.

Columns: task_id, y_a, s_b, novelty, novelty_max, diverged_at (-1 = never), y_b (optional).
`swapcert estimate` and `exp/resample.py` consume this table.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from swapcert.fork.prefix import Agent, Simulator, actions_match, default_is_stop, fork_run
from swapcert.novelty.knn import NoveltyIndex
from swapcert.traces.schema import Event, Trajectory

COLUMNS = ["task_id", "y_a", "s_b", "novelty", "novelty_max", "diverged_at", "y_b"]


def simulate_table(
    tasks: Sequence,
    a_runs: Sequence[Trajectory],
    agent_b: Agent,
    make_sim: Callable[[object], Simulator],
    novelty: NoveltyIndex,
    k: int = 3,
    seed_base: int = 2000,
    matcher: Callable[[Event, Event], bool] = actions_match,
    is_stop: Callable[[Event], bool] = default_is_stop,
    y_b: Sequence[float] | None = None,
) -> list[dict]:
    """`make_sim(task)` returns a fresh simulator per task (so state never leaks across tasks)."""
    rows = []
    for i, (t, ra) in enumerate(zip(tasks, a_runs, strict=True)):
        sim = make_sim(t)
        outs, nm, nx, div = [], [], [], -1
        for j in range(k):
            fr = fork_run(t, ra, agent_b, sim, seed=seed_base + j, matcher=matcher, is_stop=is_stop)
            outs.append(fr.s_b)
            m, x = novelty.task_novelty(fr.sim_steps)
            nm.append(m)
            nx.append(x)
            if j == 0 and fr.diverged_at is not None:
                div = fr.diverged_at
        rows.append({"task_id": t.task_id, "y_a": float(ra.success or 0.0), "s_b": float(np.mean(outs)),
                     "novelty": float(np.mean(nm)), "novelty_max": float(np.mean(nx)), "diverged_at": div,
                     "y_b": "" if y_b is None else float(y_b[i])})
    return rows


def write_table(rows: list[dict], path: str | Path, n_real: int | None = None, seed: int = 0) -> None:
    """Write the CSV; if n_real is given, keep y_b for a uniformly random subset of that size
    (the 'real run' budget) and blank it elsewhere."""
    keep = set(range(len(rows)))
    if n_real is not None:
        keep = set(np.random.default_rng(seed).choice(len(rows), min(n_real, len(rows)), replace=False).tolist())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({**r, "y_b": r["y_b"] if i in keep else ""})
