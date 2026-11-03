import csv
import json

import numpy as np
import pytest
from typer.testing import CliRunner

from swapcert.cli import app
from swapcert.envs.tau2_adapter import load_results
from swapcert.envs.toy import MockAgent, ToyEnv, make_tasks, run_real
from swapcert.exp.dataset import simulate_table
from swapcert.fork.match import Matcher, hashed_ngram_cosine
from swapcert.fork.prefix import fork_run
from swapcert.judge.outcome import OutcomeJudge, calibrate
from swapcert.llm.cache import Cache
from swapcert.llm.client import LLM
from swapcert.novelty.knn import NoveltyIndex
from swapcert.sim.kb import build_worldbook, nearest_examples
from swapcert.sim.llm_tool_sim import LLMToolSim
from swapcert.traces.schema import Event
from tests.test_bench import FIX


def scripted_llm(task, calls=None):
    """Mock LLM that plays the toy backend from the task's TRUE world (stand-in for a perfect
    simulator); judge prompts get YES if every cancel the transcript shows hit the right orders."""
    env = ToyEnv(task)

    def backend(model, messages, params):
        p = messages[0]["content"]
        if calls is not None:
            calls.append(p)
        if "OUTPUT:" in p:
            name = p.split("TOOL: ")[1].split("\n")[0]
            args = json.loads(p.split("ARGS: ")[1].split("\n")[0])
            return {"text": env.step(name, args), "in": 1, "out": 1}
        return {"text": "YES" if env.success() else "NO", "in": 1, "out": 1}

    return LLM("mock", cache=Cache(), backend=backend)


def toy_world(n=40):
    ts = make_tasks(n)
    a = MockAgent("A", 0.5)
    return ts, [run_real(t, a, seed=1000) for t in ts], MockAgent("B", 0.4, style=1)


def test_matcher_text_threshold_and_strict():
    m = Matcher(threshold=0.9)
    a = Event("assistant", content="Your order is pending. Shall I cancel it?")
    assert m(a, Event("assistant", content="Your order is pending.  Shall I cancel it? "))
    assert not m(a, Event("assistant", content="Refunds take five to seven business days."))
    assert not Matcher(strict=True)(a, Event("assistant", content="Your order is pending. Shall I cancel it"))
    assert hashed_ngram_cosine("abc def", "abc def") == pytest.approx(1.0)


def test_matcher_tool_calls_need_identical_args():
    m = Matcher()
    c1 = Event("tool_call", name="get", args={"id": "1", "x": 2})
    assert m(c1, Event("tool_call", name="get", args={"x": 2, "id": "1"}))
    assert not m(c1, Event("tool_call", name="get", args={"id": "2", "x": 2}))
    assert not m(c1, Event("assistant", content="get"))


def test_worldbook_builds_and_refuses_forbidden_model():
    _, a_runs, _ = toy_world()
    kb = build_worldbook(a_runs, forbid_model="B")
    assert set(kb.tools) == {"get_order", "cancel_order"} and kb.tools["get_order"].examples
    assert nearest_examples(kb.tools["get_order"], {"id": "O1-0"}, 2)
    with pytest.raises(ValueError, match="forbidden model"):
        build_worldbook(a_runs, forbid_model="A")


def test_worldbook_on_tau2_fixture_detects_json_and_errors():
    kb = build_worldbook(load_results(FIX, "m"))
    assert kb.tools["get_order_details"].result_kind == "json"
    assert kb.tools["get_order_details"].error_examples == ["Error: not found"]


def test_llm_sim_end_to_end_and_cache_makes_rerun_free():
    ts, a_runs, b = toy_world(30)
    kb = build_worldbook(a_runs)
    llm_calls = []

    def run_all(shared_cache):
        rows = []
        for t, ra in zip(ts, a_runs, strict=True):
            llm = scripted_llm(t, llm_calls)
            llm.cache = shared_cache
            rows += simulate_table([t], [ra], b, lambda _t, llm=llm: LLMToolSim(llm, kb), NoveltyIndex(a_runs), k=2)
        return rows

    cache = Cache()
    rows = run_all(cache)
    n_first = len(llm_calls)
    assert n_first > 0 and len(rows) == 30
    assert all(0.0 <= r["s_b"] <= 1.0 and r["diverged_at"] >= -1 for r in rows)
    run_all(cache)  # identical prompts -> served from cache, zero new backend calls
    assert len(llm_calls) == n_first


def test_perfect_mock_sim_matches_real_outcomes_on_diverged_tasks():
    ts, a_runs, b = toy_world(60)
    kb = build_worldbook(a_runs)
    agree = tot = 0
    for t, ra in zip(ts, a_runs, strict=True):
        llm = scripted_llm(t)
        fr = fork_run(t, ra, b, LLMToolSim(llm, kb), seed=2000)
        real = run_real(t, b, seed=2000)
        if fr.diverged_at is not None:
            tot += 1
            agree += (fr.s_b >= 0.5) == (real.success >= 0.5)
    assert tot > 20 and agree / tot > 0.9  # with a perfect backend, forking reproduces real outcomes


def test_malformed_reply_falls_back_and_is_counted():
    ts, a_runs, _b = toy_world(10)
    kb = build_worldbook(a_runs)
    bad = LLM("mock", cache=Cache(), backend=lambda m, msgs, p: {"text": "", "in": 1, "out": 1})
    sim = LLMToolSim(bad, kb, judge=lambda task, ev: 0.0)
    sim.start(ts[0], [])
    out = sim.step("get_order", {"id": "O0-0"})
    assert sim.health.fallbacks == 1 and out in {r for _, r in kb.tools["get_order"].examples}
    assert sim.step("nonexistent", {}).startswith("Error") and sim.health.unknown_tool == 1


def test_judge_parse_failures_and_calibration():
    ts, a_runs, _ = toy_world(12)
    llm = LLM("mock", cache=Cache(), backend=lambda m, msgs, p: {"text": "unsure", "in": 1, "out": 1})
    j = OutcomeJudge(llm)
    assert j(ts[0], a_runs[0].events) == 0.0 and j.stats.parse_failures == 1
    yes = OutcomeJudge(LLM("mock", cache=Cache(), backend=lambda m, msgs, p: {"text": "Yes.", "in": 1, "out": 1}))
    cal = calibrate(yes, {t.task_id: t for t in ts}, a_runs)
    assert cal["n"] == 12 and cal["accuracy"] == pytest.approx(np.mean([r.success for r in a_runs]))


def test_novelty_zero_on_seen_and_positive_on_unseen():
    _, a_runs, _ = toy_world()
    nov = NoveltyIndex(a_runs)
    assert nov.step_novelty("get_order", "<start>") == pytest.approx(0.0, abs=1e-9)
    assert nov.step_novelty("refund_everything", "teleport") > 0.5
    assert nov.task_novelty([]) == (0.0, 0.0)


def test_text_turn_after_fork_ends_episode_without_user_reply():
    ts, a_runs, _ = toy_world(3)

    class Chatty:
        def act(self, task, history, seed):
            return Event("assistant", content="Hello! How can I help?")

    kb = build_worldbook(a_runs)
    fr = fork_run(ts[0], a_runs[0], Chatty(), LLMToolSim(scripted_llm(ts[0]), kb), seed=1)
    assert fr.diverged_at == 0 and fr.traj.events[-1].kind == "assistant"


def test_simulate_then_estimate_cli_roundtrip(tmp_path):
    r = CliRunner()
    out = tmp_path / "t.csv"
    res = r.invoke(app, ["simulate", str(out), "--toy", "--tasks", "120", "--n-real", "50"])
    assert res.exit_code == 0, res.output
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 120 and sum(bool(x["y_b"]) for x in rows) == 50
    est = r.invoke(app, ["estimate", str(out), "--tolerance", "0.05"])
    assert est.exit_code == 0, est.output
    assert "Verdict:" in est.output


def test_call_result_pairs_handles_parallel_calls():
    from swapcert.traces.schema import call_result_pairs
    ev = [Event("tool_call", name="a", args={"i": 1}), Event("tool_call", name="b", args={"i": 2}),
          Event("tool_result", content="rb", name="b"), Event("tool_result", content="ra", name="a")]
    assert [(c.name, r.content) for c, r in call_result_pairs(ev)] == [("b", "rb"), ("a", "ra")]
