"""Worldbook: what a simulator may know about the environment, learned from A's traces only."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

from swapcert.traces.schema import Trajectory, call_result_pairs


@dataclass
class ToyEntry:
    name: str
    n_calls: int = 0
    arg_names: list[str] = field(default_factory=list)
    examples: list[tuple[dict, str]] = field(default_factory=list)  # (args, result)
    error_examples: list[str] = field(default_factory=list)
    result_kind: str = "text"  # "json" if most results parse as JSON


ToolEntry = ToyEntry  # public alias


@dataclass
class Worldbook:
    tools: dict[str, ToolEntry]
    n_trajectories: int

    def describe(self, name: str) -> str:
        t = self.tools[name]
        return (f"TOOL {t.name}: args={t.arg_names}; results are {t.result_kind}; "
                f"seen {t.n_calls} times")


def _is_error(s: str) -> bool:
    return bool(re.match(r"\s*(error|exception|invalid)", s, re.IGNORECASE))


def _tokens(args: dict) -> set[str]:
    return {f"{k}={v}" for k, v in args.items()} | {f"{k}" for k in args}


def build_worldbook(trajs: list[Trajectory], forbid_model: str | None = None, k_examples: int = 5) -> Worldbook:
    """Build from recorded runs.  `forbid_model` guards against building the simulator from the
    candidate model's own traces (the simulator must never see B).  For an A->A control pass None."""
    if forbid_model is not None:
        bad = [t for t in trajs if t.model == forbid_model]
        if bad:
            raise ValueError(f"{len(bad)} trajectories come from forbidden model {forbid_model!r}")
    tools: dict[str, ToolEntry] = {}
    kinds: dict[str, Counter] = {}
    for t in trajs:
        for e, r in call_result_pairs(t.events):
            ent = tools.setdefault(e.name, ToolEntry(e.name))
            res = r.content
            ent.n_calls += 1
            for k in (e.args or {}):
                if k not in ent.arg_names:
                    ent.arg_names.append(k)
            pair = (e.args or {}, res)
            if _is_error(res):
                if res not in ent.error_examples and len(ent.error_examples) < k_examples:
                    ent.error_examples.append(res)
            else:
                if pair not in ent.examples and len(ent.examples) < k_examples * 4:
                    ent.examples.append(pair)
                try:  # only successful results vote on the format; error strings are always text
                    json.loads(res)
                    kinds.setdefault(e.name, Counter())["json"] += 1
                except (ValueError, TypeError):
                    kinds.setdefault(e.name, Counter())["text"] += 1
    for name, ent in tools.items():
        ent.result_kind = "json" if kinds.get(name, Counter())["json"] > kinds.get(name, Counter())["text"] else "text"
    return Worldbook(tools, len(trajs))


def nearest_examples(ent: ToolEntry, args: dict, k: int = 3) -> list[tuple[dict, str]]:
    """Deterministic retrieval: Jaccard overlap on arg key=value tokens, ties by insertion order."""
    q = _tokens(args)
    scored = sorted(enumerate(ent.examples), key=lambda ie: (-len(q & _tokens(ie[1][0])) / max(len(q | _tokens(ie[1][0])), 1), ie[0]))
    return [ex for _, ex in scored[:k]]
