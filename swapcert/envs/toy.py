"""A small deterministic stateful environment ("orders DB") plus stochastic mock agents.

Purpose: exercise forking, simulation, judging and the full pipeline end-to-end with no LLM.
It is NOT evidence about real agents.

Task: cancel a set of target orders (only 'pending' ones can be cancelled) without touching others.
Tools: get_order(id) -> status ; cancel_order(id) -> ok | error ; finish().
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from swapcert.traces.schema import Event, Trajectory


@dataclass
class Task:
    task_id: str
    orders: dict[str, str]  # id -> status
    targets: list[str]  # ids to cancel (some may already be shipped -> cannot be cancelled)

    def expected_final(self) -> dict[str, str]:
        return {o: ("cancelled" if o in self.targets and st == "pending" else st) for o, st in self.orders.items()}

    def goal_met(self, final: dict[str, str]) -> bool:
        return all(final[o] == want for o, want in self.expected_final().items())


def make_tasks(n: int, seed: int = 0) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        k = rng.randint(2, 5)
        ids = [f"O{i}-{j}" for j in range(k)]
        orders = {o: rng.choice(["pending", "pending", "shipped"]) for o in ids}
        targets = rng.sample(ids, rng.randint(1, min(2, k)))
        tasks.append(Task(f"toy-{i}", orders, targets))
    return tasks


class ToyEnv:
    def __init__(self, task: Task):
        self.task = task
        self.state = dict(task.orders)

    def step(self, name: str, args: dict) -> str:
        oid = args.get("id")
        if name == "get_order":
            return self.state.get(oid, "unknown")
        if name == "cancel_order":
            if self.state.get(oid) == "pending":
                self.state[oid] = "cancelled"
                return "ok"
            return f"error:{self.state.get(oid, 'unknown')}"
        return "error:unknown_tool"

    def success(self) -> float:
        return float(self.task.goal_met(self.state))


@dataclass
class MockAgent:
    """Stochastic agent. `skill` in [0,1] drives carefulness; `style` changes the action sequence
    (so two equally skilled agents still diverge, like different model families do)."""

    name: str
    skill: float
    style: int = 0
    max_steps: int = 14

    def act(self, task: Task, history: list[Event], seed: int) -> Event:
        """Next action given history. Deterministic in (task, history length, seed, agent)."""
        h = hashlib.sha256(f"{self.name}|{task.task_id}|{len(history)}|{seed}".encode()).digest()
        rng = random.Random(int.from_bytes(h[:8], "big"))
        done_cancel = {e.args["id"] for e in history if e.kind == "tool_call" and e.name == "cancel_order"}
        checked = {e.args["id"]: history[i + 1].content for i, e in enumerate(history[:-1])
                   if e.kind == "tool_call" and e.name == "get_order"}
        order = list(reversed(task.targets)) if self.style == 1 else task.targets
        for oid in order:
            if oid in done_cancel:
                continue
            careful = rng.random() < self.skill
            if not careful and rng.random() < 0.6 * (1 - self.skill):
                return Event("tool_call", name="finish", args={})  # gives up early
            if careful and oid not in checked:
                return Event("tool_call", name="get_order", args={"id": oid})
            if careful and checked.get(oid) != "pending":
                done_cancel.add(oid)  # skip: cannot cancel
                continue
            target = oid
            if not careful and rng.random() < 0.5 * (1 - self.skill) + 0.1:
                target = rng.choice(list(task.orders))  # slip: cancels the wrong order
            return Event("tool_call", name="cancel_order", args={"id": target})
        return Event("tool_call", name="finish", args={})


def run_real(task: Task, agent: MockAgent, seed: int, model: str | None = None) -> Trajectory:
    env, hist = ToyEnv(task), []
    for _ in range(agent.max_steps):
        a = agent.act(task, hist, seed)
        hist.append(a)
        if a.name == "finish":
            break
        hist.append(Event("tool_result", content=env.step(a.name, a.args or {}), name=a.name))
    return Trajectory(task.task_id, model or agent.name, hist, env.success(), trial=seed)
