"""Exact-prefix forking.

Replay A's recorded trajectory event-for-event while B's proposed action matches A's.  The first
time B acts differently, the recorded world no longer applies, so control passes to a simulator.
Up to the fork, outcomes are exactly what the real environment produced; after it, they are
simulated.  If B never diverges, A's recorded outcome is B's outcome (no simulation needed).

Generalised for tau2-shaped traces: after a matched agent action we replay *all* recorded
non-agent events that follow it (tool results, user turns) up to the next agent action.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from swapcert.traces.schema import Event, Trajectory


class Agent(Protocol):
    def act(self, task, history: list[Event], seed: int) -> Event: ...


class Simulator(Protocol):
    """`reply(text)` is optional: it returns the simulated user's answer to an assistant message.
    Simulators without it end the episode when the agent addresses the user after the fork."""

    def start(self, task, prefix: list[Event]) -> None: ...
    def step(self, name: str, args: dict) -> str: ...
    def judge(self) -> float: ...


@dataclass
class ForkResult:
    traj: Trajectory
    diverged_at: int | None  # index into the event list where B first differed; None = never
    s_b: float  # estimated success of B on this task (recorded outcome if never diverged)
    sim_steps: list[tuple[str, str]]  # (tool, prev_tool) bigrams visited after the fork


def actions_match(a: Event, b: Event) -> bool:
    return a.canonical_action() == b.canonical_action()


def default_is_stop(e: Event) -> bool:
    return e.name == "finish"


def _recorded_observations(traj: Trajectory, start: int) -> list[Event]:
    """Recorded non-agent events from `start` until the next agent action (or the end)."""
    out = []
    for e in traj.events[start:]:
        if e.kind in ("tool_call", "assistant"):
            break
        out.append(e)
    return out


def fork_run(
    task,
    traj_a: Trajectory,
    agent_b: Agent,
    sim: Simulator,
    seed: int,
    max_steps: int = 14,
    matcher: Callable[[Event, Event], bool] = actions_match,
    is_stop: Callable[[Event], bool] = default_is_stop,
) -> ForkResult:
    hist: list[Event] = []
    diverged_at: int | None = None
    sim_steps: list[tuple[str, str]] = []
    prev = "<start>"
    for _ in range(max_steps):
        a = agent_b.act(task, hist, seed)
        pos = len(hist)
        if diverged_at is None:
            rec = traj_a.events[pos] if pos < len(traj_a.events) else None
            if rec is not None and rec.kind in ("tool_call", "assistant") and matcher(a, rec):
                hist.append(a)
                if is_stop(a):
                    break
                hist.extend(_recorded_observations(traj_a, pos + 1))
                prev = a.name or prev
                continue
            diverged_at = pos
            sim.start(task, hist)
        hist.append(a)
        if is_stop(a):
            break
        if a.kind == "tool_call":
            sim_steps.append((a.name or "", prev))
            hist.append(Event("tool_result", content=sim.step(a.name or "", a.args or {}), name=a.name))
            prev = a.name or prev
        else:  # assistant text addressed to the user
            reply = getattr(sim, "reply", None)
            if reply is None:
                break
            hist.append(Event("user", content=reply(a.content)))
    if hasattr(sim, "observe") and diverged_at is not None:
        sim.observe(hist)
    s_b = traj_a.success if diverged_at is None else sim.judge()
    out = Trajectory(task.task_id, "forked", hist, s_b, trial=seed, meta={"diverged_at": diverged_at})
    return ForkResult(out, diverged_at, float(s_b if s_b is not None else 0.0), sim_steps)
