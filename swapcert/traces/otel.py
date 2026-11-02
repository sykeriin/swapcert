"""Import/export trajectories as OpenTelemetry GenAI-style span records.

One trajectory = one root span (`invoke_agent`) with child spans in order:
  - `chat`          assistant/user text            (gen_ai.operation.name = chat)
  - `execute_tool`  tool call + result             (gen_ai.operation.name = execute_tool)
Attribute names follow the GenAI semantic conventions (gen_ai.*); the conventions are still
marked experimental, so unknown attributes are preserved under `swapcert.*` and ignored on read.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .schema import Event, Trajectory


def to_otel(traj: Trajectory) -> list[dict[str, Any]]:
    trace_id = hashlib.sha256(f"{traj.task_id}:{traj.trial}".encode()).hexdigest()[:16]
    root = {
        "trace_id": trace_id,
        "span_id": "0",
        "parent_span_id": None,
        "name": "invoke_agent",
        "attributes": {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.request.model": traj.model,
            "swapcert.task_id": traj.task_id,
            "swapcert.trial": traj.trial,
            "swapcert.success": traj.success,
        },
    }
    spans = [root]
    i, n, sid = 0, len(traj.events), 1
    while i < n:
        e = traj.events[i]
        if e.kind == "tool_call":
            res = traj.events[i + 1] if i + 1 < n and traj.events[i + 1].kind == "tool_result" else None
            attrs = {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": e.name,
                "gen_ai.tool.call.arguments": json.dumps(e.args or {}, sort_keys=True),
            }
            if res is not None:
                attrs["gen_ai.tool.call.result"] = res.content
            i += 2 if res is not None else 1
        else:
            attrs = {
                "gen_ai.operation.name": "chat",
                "gen_ai.message.role": e.kind,
                "gen_ai.message.content": e.content,
            }
            i += 1
        spans.append(
            {"trace_id": trace_id, "span_id": str(sid), "parent_span_id": "0", "name": attrs["gen_ai.operation.name"], "attributes": attrs}
        )
        sid += 1
    return spans


def from_otel(spans: list[dict[str, Any]]) -> Trajectory:
    root = next(s for s in spans if s["parent_span_id"] is None)
    ra = root["attributes"]
    traj = Trajectory(
        task_id=ra["swapcert.task_id"],
        model=ra["gen_ai.request.model"],
        success=ra.get("swapcert.success"),
        trial=ra.get("swapcert.trial", 0),
    )
    children = sorted((s for s in spans if s["parent_span_id"] == "0"), key=lambda s: int(s["span_id"]))
    for s in children:
        a = s["attributes"]
        if a["gen_ai.operation.name"] == "execute_tool":
            traj.events.append(Event("tool_call", name=a["gen_ai.tool.name"], args=json.loads(a["gen_ai.tool.call.arguments"])))
            if "gen_ai.tool.call.result" in a:
                traj.events.append(Event("tool_result", content=a["gen_ai.tool.call.result"], name=a["gen_ai.tool.name"]))
        else:
            traj.events.append(Event(a["gen_ai.message.role"], content=a["gen_ai.message.content"]))
    return traj
