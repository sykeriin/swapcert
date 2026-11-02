"""Trace-built simulator for the toy env (stand-in for the LLM-backed simulator used on τ²).

Built ONLY from model A's traces: it learns the tool response format and the base rate of
'pending' orders.  It sees nothing about B.  `error` injects simulator infidelity so tests can
sweep proxy quality.
"""

from __future__ import annotations

import random
from collections import Counter

from swapcert.traces.schema import Event, Trajectory


class ToySim:
    def __init__(self, a_traces: list[Trajectory], error: float = 0.1, seed: int = 0):
        c = Counter(e.content for t in a_traces for i, e in enumerate(t.events)
                    if e.kind == "tool_result" and t.events[i - 1].name == "get_order")
        tot = sum(c.values()) or 1
        self.p_pending = c["pending"] / tot
        self.error, self.rng = error, random.Random(seed)
        self.state: dict[str, str] = {}
        self.task = None

    def start(self, task, prefix: list[Event]) -> None:
        self.task, self.state = task, {}
        for i, e in enumerate(prefix[:-1]):  # replay the world implied by the recorded prefix
            if e.kind == "tool_call" and e.name == "get_order":
                self.state[e.args["id"]] = prefix[i + 1].content
            elif e.kind == "tool_call" and e.name == "cancel_order" and prefix[i + 1].content == "ok":
                self.state[e.args["id"]] = "cancelled"

    def _belief(self, oid: str) -> str:
        if oid not in self.state:
            self.state[oid] = "pending" if self.rng.random() < self.p_pending else "shipped"
        return self.state[oid]

    def step(self, name: str, args: dict) -> str:
        oid = args.get("id", "")
        if name == "get_order":
            out = self._belief(oid)
        elif name == "cancel_order":
            if self._belief(oid) == "pending":
                self.state[oid], out = "cancelled", "ok"
            else:
                out = f"error:{self.state[oid]}"
        else:
            out = "error:unknown_tool"
        if self.rng.random() < self.error:  # simulator infidelity
            out = {"pending": "shipped", "shipped": "pending", "ok": "error:shipped"}.get(out, out)
        return out

    def judge(self) -> float:
        """Success judged ONLY from the simulator's beliefs (no peeking at true order statuses):
        every target is cancelled or observed to be un-cancellable; no non-target was cancelled."""
        t = self.task
        for oid in t.orders:
            st = self.state.get(oid)
            if oid in t.targets and st not in ("cancelled", "shipped"):
                return 0.0
            if oid not in t.targets and st == "cancelled":
                return 0.0
        return 1.0
