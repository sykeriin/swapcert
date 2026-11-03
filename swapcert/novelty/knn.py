"""Out-of-trace novelty: how far is a simulated step from anything A actually did?

Step features = hashed tokens of (tool, previous tool, tool|previous).  Novelty of a step is
1 - max cosine similarity to the recorded steps; 0 = seen before, 1 = nothing similar.
Deterministic (crc32 hashing), so no embedding model is needed for tests.
"""

from __future__ import annotations

import math
import zlib

from swapcert.traces.schema import Trajectory

DIM = 256


def _vec(tool: str, prev: str) -> dict[int, float]:
    v: dict[int, float] = {}
    for tok in (f"t={tool}", f"p={prev}", f"tp={tool}|{prev}"):
        k = zlib.crc32(tok.encode()) % DIM
        v[k] = v.get(k, 0.0) + 1.0
    n = math.sqrt(sum(x * x for x in v.values()))
    return {k: x / n for k, x in v.items()}


def _cos(a: dict[int, float], b: dict[int, float]) -> float:
    return sum(x * b.get(k, 0.0) for k, x in a.items())


class NoveltyIndex:
    def __init__(self, recorded: list[Trajectory]):
        seen: set[tuple[str, str]] = set()
        for t in recorded:
            prev = "<start>"
            for e in t.events:
                if e.kind == "tool_call":
                    seen.add((e.name or "", prev))
                    prev = e.name or prev
        self._ref = [_vec(*s) for s in sorted(seen)]

    def step_novelty(self, tool: str, prev: str) -> float:
        if not self._ref:
            return 1.0
        v = _vec(tool, prev)
        return max(0.0, 1.0 - max(_cos(v, r) for r in self._ref))

    def task_novelty(self, steps: list[tuple[str, str]]) -> tuple[float, float]:
        """(mean, max) novelty over simulated steps; (0, 0) if the run never left the recording."""
        if not steps:
            return 0.0, 0.0
        n = [self.step_novelty(t, p) for t, p in steps]
        return sum(n) / len(n), max(n)
