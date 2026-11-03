"""Convert τ²-bench result files into swapcert Trajectories.

Schema source: tau2-bench @ 19f13ea (src/tau2/data_model/{simulation,message}.py), read from
source -- NOT yet validated against a real run (tau2 could not be installed in the build
container).  Validate on the laptop with `swapcert import-tau2` and fix the fixture if it differs.

Message mapping
  assistant  content            -> Event(assistant)
  assistant  tool_calls[i]      -> Event(tool_call, name, args)            (agent action)
  user       content            -> Event(user)
  user       tool_calls[i]      -> Event(user, name=<tool>, args=<args>)   (environment action by the
                                   simulated user; not an agent action, so never compared in forking)
  tool       (requestor=*)      -> Event(tool_result, content, name)       name resolved by call id
  tool       tool_messages=[..] -> one tool_result per inner message       (MultiToolMessage)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from swapcert.traces.schema import Event, Trajectory


def _events_from_messages(messages: list[dict[str, Any]]) -> list[Event]:
    events: list[Event] = []
    call_names: dict[str, str] = {}

    def tool_result(m: dict[str, Any]) -> None:
        events.append(Event("tool_result", content=m.get("content") or "", name=call_names.get(m.get("id", ""))))

    for m in messages:
        role = m.get("role")
        if role == "assistant":
            if m.get("content"):
                events.append(Event("assistant", content=m["content"]))
            for tc in m.get("tool_calls") or []:
                call_names[tc.get("id", "")] = tc["name"]
                events.append(Event("tool_call", name=tc["name"], args=tc.get("arguments") or {}))
        elif role == "user":
            if m.get("content"):
                events.append(Event("user", content=m["content"]))
            for tc in m.get("tool_calls") or []:
                call_names[tc.get("id", "")] = tc["name"]
                events.append(Event("user", name=tc["name"], args=tc.get("arguments") or {}))
        elif role == "tool":
            for inner in m.get("tool_messages") or [m]:
                tool_result(inner)
        # system messages are not part of the interaction trace
    return events


def trajectory_from_simulation(sim: dict[str, Any], model: str) -> Trajectory:
    reward = (sim.get("reward_info") or {}).get("reward")
    return Trajectory(
        task_id=str(sim["task_id"]),
        model=model,
        events=_events_from_messages(sim.get("messages") or []),
        success=None if reward is None else float(reward),
        trial=int(sim.get("trial") or 0),
        meta={
            "termination_reason": sim.get("termination_reason"),
            "duration": sim.get("duration"),
            "seed": sim.get("seed"),
        },
    )


def load_results(path: str | Path, model: str) -> list[Trajectory]:
    """Read a results.json (single-file layout) or a results directory (dir layout)."""
    p = Path(path)
    if p.is_dir():
        p = p / "results.json"
    data = json.loads(p.read_text())
    sims = data.get("simulations")
    if not sims:  # dir layout: one JSON file per simulation next to results.json
        sim_dir = p.parent / "simulations"
        sims = [json.loads(f.read_text()) for f in sorted(sim_dir.glob("*.json"))] if sim_dir.is_dir() else []
    return [trajectory_from_simulation(s, model) for s in sims]


def terminal_error_rate(trajs: list[Trajectory]) -> float:
    """Share of runs that ended in an error-type termination (gate G3 in the plan)."""
    bad = {"agent_error", "user_error", "infrastructure_error", "too_many_errors",
           "context_window_exceeded", "unexpected_error"}
    if not trajs:
        return 0.0
    return sum(t.meta.get("termination_reason") in bad for t in trajs) / len(trajs)
