"""Action-equivalence for forking.

Tool calls match on name + canonical JSON args.  Assistant text matches only if a similarity
function says the two messages are near-identical (default: deterministic character-trigram
cosine, no model needed; on the laptop inject an embedding-based similarity).  `strict=True`
requires identical text and is the ablation from the plan (§4.4).
"""

from __future__ import annotations

import math
import zlib
from collections import Counter
from collections.abc import Callable

from swapcert.traces.schema import Event

Similarity = Callable[[str, str], float]


def hashed_ngram_cosine(a: str, b: str, n: int = 3, dim: int = 2048) -> float:
    """Cosine similarity of hashed character n-gram count vectors (deterministic across runs)."""

    def vec(s: str) -> Counter:
        s = " ".join(s.lower().split())
        return Counter(zlib.crc32(s[i : i + n].encode()) % dim for i in range(max(len(s) - n + 1, 1)))

    va, vb = vec(a), vec(b)
    dot = sum(c * vb.get(k, 0) for k, c in va.items())
    na = math.sqrt(sum(c * c for c in va.values()))
    nb = math.sqrt(sum(c * c for c in vb.values()))
    return dot / (na * nb) if na and nb else 0.0


class Matcher:
    def __init__(self, text_sim: Similarity | None = hashed_ngram_cosine, threshold: float = 0.9,
                 strict: bool = False):
        self.text_sim, self.threshold, self.strict = text_sim, threshold, strict

    def __call__(self, b_action: Event, recorded: Event) -> bool:
        if b_action.kind == "tool_call" and recorded.kind == "tool_call":
            return b_action.canonical_action() == recorded.canonical_action()
        if b_action.kind == "assistant" and recorded.kind == "assistant":
            if self.strict or self.text_sim is None:
                return b_action.content.strip() == recorded.content.strip()
            return self.text_sim(b_action.content, recorded.content) >= self.threshold
        return False
