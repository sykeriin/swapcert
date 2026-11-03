import json
from pathlib import Path

from swapcert.envs.tau2_adapter import load_results, terminal_error_rate
from swapcert.traces.otel import from_otel, to_otel

FIX = Path(__file__).parent / "fixtures" / "tau2_results_sample.json"


def test_load_single_file_layout():
    ts = load_results(FIX, model="ollama_chat/qwen3:4b")
    assert [t.task_id for t in ts] == ["0", "1"]
    assert ts[0].success == 1.0 and ts[1].success == 0.0
    acts = [e.canonical_action() for e in ts[0].events if e.canonical_action()]
    assert acts[0].startswith("call:get_order_details:") and "say:Your order is pending" in acts[1]


def test_tool_results_resolve_call_names_including_multitool():
    ts = load_results(FIX, model="m")
    res = [e for e in ts[1].events if e.kind == "tool_result"]
    assert [(e.name, e.content) for e in res] == [("get_order_details", "Error: not found"),
                                                  ("get_user_details", '{"name": "Ann"}')]


def test_null_assistant_content_emits_no_empty_text_event():
    ts = load_results(FIX, model="m")
    assert all(e.content for e in ts[0].events if e.kind in ("assistant", "user") and e.name is None)


def test_dir_layout(tmp_path):
    data = json.loads(FIX.read_text())
    sims = data.pop("simulations")
    (tmp_path / "simulations").mkdir()
    for i, s in enumerate(sims):
        (tmp_path / "simulations" / f"sim{i}.json").write_text(json.dumps(s))
    (tmp_path / "results.json").write_text(json.dumps(data))
    assert [t.task_id for t in load_results(tmp_path, "m")] == ["0", "1"]


def test_error_rate_and_otel_roundtrip():
    ts = load_results(FIX, model="m")
    assert terminal_error_rate(ts) == 0.5
    back = from_otel(to_otel(ts[0]))
    assert [e.canonical_action() for e in back.events] == [e.canonical_action() for e in ts[0].events]
