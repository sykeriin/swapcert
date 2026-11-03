"""LLM-backed tool-backend simulator (Trace2Env-style: worldbook + episodic state).

The environment's real database is NOT visible.  The simulator knows (a) the worldbook built from
A's traces, (b) the session so far (prefix replayed from the recording + its own earlier answers).
Malformed or empty replies fall back to the nearest recorded example and are counted -- those
counters are the simulator-health metric to report.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from swapcert.judge.outcome import OutcomeJudge
from swapcert.llm.client import LLM
from swapcert.sim.kb import Worldbook, nearest_examples
from swapcert.traces.schema import Event, call_result_pairs


@dataclass
class SimHealth:
    calls: int = 0
    fallbacks: int = 0
    unknown_tool: int = 0


class LLMToolSim:
    def __init__(self, llm: LLM, kb: Worldbook, judge: OutcomeJudge | Callable | None = None,
                 k_examples: int = 3, max_history: int = 30):
        self.llm, self.kb, self.k, self.max_history = llm, kb, k_examples, max_history
        self.judge_fn = judge or OutcomeJudge(llm)
        self.health = SimHealth()
        self.task = None
        self.session: list[tuple[str, dict, str]] = []
        self.transcript: list[Event] = []

    def start(self, task, prefix: list[Event]) -> None:
        self.task, self.session, self.transcript = task, [], list(prefix)
        for c, r in call_result_pairs(prefix):  # episodic state = what the recorded prefix revealed
            self.session.append((c.name, c.args or {}, r.content))

    def step(self, name: str, args: dict) -> str:
        self.health.calls += 1
        if name not in self.kb.tools:
            self.health.unknown_tool += 1
            out = f"Error: unknown tool {name}"
        else:
            out = self._ask(name, args)
        self.session.append((name, args, out))
        return out

    def _ask(self, name: str, args: dict) -> str:
        ent = self.kb.tools[name]
        ex = nearest_examples(ent, args, self.k)
        hist = "\n".join(f"{n}({a}) -> {r}" for n, a, r in self.session[-self.max_history :]) or "(none yet)"
        prompt = (
            "You simulate a backend API for a customer-service agent. Reply with ONLY the raw tool output, "
            "no commentary. Stay consistent with the session so far.\n"
            f"{self.kb.describe(name)}\n"
            "Examples of real outputs:\n" + "\n".join(f"{name}({a}) -> {r}" for a, r in ex) + "\n"
            + ("Example errors:\n" + "\n".join(ent.error_examples[:2]) + "\n" if ent.error_examples else "")
            + f"Session so far:\n{hist}\n"
            f"TOOL: {name}\nARGS: {json.dumps(args, sort_keys=True)}\nOUTPUT:"
        )
        out = self.llm.complete([{"role": "user", "content": prompt}], temperature=0.0).strip()
        if not out or (ent.result_kind == "json" and not _parses(out) and not out.lower().startswith("error")):
            self.health.fallbacks += 1
            return ex[0][1] if ex else "Error: no data"
        return out

    def observe(self, full_history: list[Event]) -> None:
        self.transcript = full_history

    def judge(self) -> float:
        return float(self.judge_fn(self.task, self.transcript))


def _parses(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except ValueError:
        return False
