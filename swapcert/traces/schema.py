"""Canonical trajectory format used everywhere inside swapcert."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Kind = Literal["user", "assistant", "tool_call", "tool_result"]


@dataclass
class Event:
    kind: Kind
    content: str = ""  # text for user/assistant/tool_result
    name: str | None = None  # tool name for tool_call / tool_result
    args: dict[str, Any] | None = None  # tool arguments for tool_call

    def canonical_action(self) -> str | None:
        """Comparable form of an agent action, or None if this event is not an agent action."""
        if self.kind == "tool_call":
            return f"call:{self.name}:{json.dumps(self.args or {}, sort_keys=True)}"
        if self.kind == "assistant":
            return f"say:{self.content.strip()}"
        return None


@dataclass
class Trajectory:
    task_id: str
    model: str
    events: list[Event] = field(default_factory=list)
    success: float | None = None  # real outcome in [0,1]; None for unscored
    trial: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def actions(self) -> list[tuple[int, Event]]:
        return [(i, e) for i, e in enumerate(self.events) if e.kind in ("tool_call", "assistant")]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json(s: str) -> Trajectory:
        d = json.loads(s)
        d["events"] = [Event(**e) for e in d["events"]]
        return Trajectory(**d)


def write_jsonl(path: str, trajs: list[Trajectory]) -> None:
    with open(path, "w") as f:
        f.writelines(t.to_json() + "\n" for t in trajs)


def read_jsonl(path: str) -> list[Trajectory]:
    with open(path) as f:
        return [Trajectory.from_json(line) for line in f if line.strip()]


def call_result_pairs(events: list[Event]) -> list[tuple[Event, Event]]:
    """Pair each agent tool_call with its tool_result, robust to parallel calls.

    Calls waiting for a result are kept in order; a result consumes the first pending call with
    the same tool name (or simply the first pending call if the result carries no name).
    """
    pending: list[Event] = []
    pairs: list[tuple[Event, Event]] = []
    for e in events:
        if e.kind == "tool_call":
            pending.append(e)
        elif e.kind == "tool_result" and pending:
            idx = next((i for i, c in enumerate(pending) if e.name in (None, c.name)), 0)
            pairs.append((pending.pop(idx), e))
    return pairs
