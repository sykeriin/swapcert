"""End-to-end toy experiment: record A, ground-truth B, forked simulation, resampled estimators."""

from __future__ import annotations

import numpy as np

from swapcert.envs.toy import MockAgent, make_tasks, run_real
from swapcert.fork.prefix import fork_run
from swapcert.sim.toy_sim import ToySim


def build_dataset(n_tasks=278, agent_a=None, agent_b=None, sim_error=0.1, k_sim=3, trials_truth=4, seed=0):
    agent_a = agent_a or MockAgent("A", skill=0.5)
    agent_b = agent_b or MockAgent("B", skill=0.4, style=1)
    tasks = make_tasks(n_tasks, seed)
    # "production logs": one recorded run of A per task
    a_runs = [run_real(t, agent_a, seed=1000) for t in tasks]
    # the simulator is built from A's traces of an independent half only (never sees B)
    half = len(tasks) // 2
    sim = ToySim(a_runs[:half], error=sim_error, seed=seed)
    seen_bigrams = {(t.events[i].name, t.events[i - 2].name if i >= 2 else "<start>")
                    for t in a_runs for i in range(0, len(t.events), 2) if t.events[i].kind == "tool_call"}
    y_a = np.array([r.success for r in a_runs])
    # ground truth for B: mean over several exhaustive real trials
    y_b = np.array([np.mean([run_real(t, agent_b, seed=s).success for s in range(trials_truth)]) for t in tasks])
    s_b, nov, div = np.zeros(len(tasks)), np.zeros(len(tasks)), np.full(len(tasks), -1)
    for i, (t, ra) in enumerate(zip(tasks, a_runs)):
        outs, novs = [], []
        for k in range(k_sim):
            fr = fork_run(t, ra, agent_b, sim, seed=2000 + k)
            outs.append(fr.s_b)
            novs.append(np.mean([b not in seen_bigrams for b in fr.sim_steps]) if fr.sim_steps else 0.0)
            div[i] = fr.diverged_at if fr.diverged_at is not None else -1
        s_b[i], nov[i] = np.mean(outs), np.mean(novs)
    return dict(y_a=y_a, y_b=y_b, s_b=s_b, nov=nov, diverged_at=div, tasks=tasks)
