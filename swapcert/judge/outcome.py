"""LLM outcome judge for simulated trajectories: did the agent accomplish the task?"""

from __future__ import annotations

import re
from dataclasses import dataclass

from swapcert.llm.client import LLM
from swapcert.traces.schema import Event


def render_transcript(events: list[Event], max_chars: int = 6000) -> str:
    lines = []
    for e in events:
        if e.kind == "tool_call":
            lines.append(f"AGENT CALLS {e.name}({e.args})")
        elif e.kind == "tool_result":
            lines.append(f"TOOL RESULT: {e.content}")
        elif e.kind == "assistant":
            lines.append(f"AGENT: {e.content}")
        else:
            lines.append(f"USER: {e.content}" if e.name is None else f"USER TOOL {e.name}({e.args})")
    text = "\n".join(lines)
    return text[-max_chars:]  # keep the end: outcomes are decided late


@dataclass
class JudgeStats:
    calls: int = 0
    parse_failures: int = 0


class OutcomeJudge:
    def __init__(self, llm: LLM):
        self.llm, self.stats = llm, JudgeStats()

    def __call__(self, task, events: list[Event]) -> float:
        desc = getattr(task, "description", None) or getattr(task, "task_id", "")
        prompt = (f"Task given to a customer-service agent:\n{desc}\n\nTranscript:\n{render_transcript(events)}\n\n"
                  "Did the agent fully and correctly accomplish the task, without unwanted side effects? "
                  "Answer with exactly one word: YES or NO.")
        out = self.llm.complete([{"role": "user", "content": prompt}], temperature=0.0)
        self.stats.calls += 1
        m = re.search(r"\b(YES|NO)\b", out.upper())
        if m is None:
            self.stats.parse_failures += 1
            return 0.0  # unparseable counts as failure; reported via stats.parse_failures
        return 1.0 if m.group(1) == "YES" else 0.0


def calibrate(judge: OutcomeJudge, tasks_by_id: dict, trajs) -> dict:
    """Agreement of the judge with the REAL reward on A's recorded runs (before trusting it on sims)."""
    pairs = [(judge(tasks_by_id.get(t.task_id, t), t.events), t.success) for t in trajs if t.success is not None]
    n = len(pairs)
    agree = sum((j >= 0.5) == (s >= 0.5) for j, s in pairs)
    return {"n": n, "accuracy": agree / n if n else float("nan"),
            "parse_failures": judge.stats.parse_failures}
