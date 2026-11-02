import numpy as np

from swapcert.envs.toy import MockAgent, ToyEnv, make_tasks, run_real
from swapcert.fork.prefix import fork_run
from swapcert.llm.cache import Cache
from swapcert.llm.client import LLM, BudgetExceeded
from swapcert.sim.toy_sim import ToySim
from swapcert.stats.decision import reliability_warning
from swapcert.traces.otel import from_otel, to_otel
from swapcert.traces.schema import Trajectory


def test_otel_roundtrip():
    t = make_tasks(3)[1]
    traj = run_real(t, MockAgent("A", 0.7), seed=3)
    back = from_otel(to_otel(traj))
    assert back.task_id == traj.task_id and back.success == traj.success
    assert [e.canonical_action() for e in back.events] == [e.canonical_action() for e in traj.events]
    assert Trajectory.from_json(traj.to_json()).events == traj.events


def test_otel_ids_are_deterministic():
    traj = run_real(make_tasks(1)[0], MockAgent("A", 0.7), seed=1)
    assert to_otel(traj)[0]["trace_id"] == to_otel(traj)[0]["trace_id"]


def test_toy_env_rules():
    task = make_tasks(5)[0]
    env = ToyEnv(task)
    pending = next((o for o, s in task.orders.items() if s == "pending"), None)
    if pending:
        assert env.step("cancel_order", {"id": pending}) == "ok"
        assert env.step("cancel_order", {"id": pending}).startswith("error")


def test_fork_identical_agent_never_diverges_and_reproduces_outcome():
    tasks = make_tasks(30)
    a = MockAgent("A", 0.6)
    for t in tasks:
        ra = run_real(t, a, seed=5)
        sim = ToySim([ra])
        fr = fork_run(t, ra, a, sim, seed=5)  # same agent, same seed => identical actions
        assert fr.diverged_at is None and fr.s_b == ra.success


def test_fork_prefix_is_real_before_divergence():
    tasks = make_tasks(40)
    ra_agent, rb_agent = MockAgent("A", 0.6), MockAgent("B", 0.4, style=1)
    for t in tasks:
        ra = run_real(t, ra_agent, seed=5)
        fr = fork_run(t, ra, rb_agent, ToySim([ra]), seed=9)
        k = fr.diverged_at
        if k is not None:
            assert [e.canonical_action() for e in fr.traj.events[:k]] == [e.canonical_action() for e in ra.events[:k]]
            assert fr.traj.events[k].canonical_action() != (ra.events[k].canonical_action() if k < len(ra.events) else None)


def test_simulator_never_reads_b():
    # ToySim is constructed from A's traces and has no handle to any agent.
    import inspect
    assert "agent" not in " ".join(inspect.signature(ToySim.__init__).parameters)


def test_llm_cache_and_budget():
    calls = []
    backend = lambda m, msgs, p: calls.append(1) or {"text": "hi", "in": 3, "out": 1}
    llm = LLM("m", cache=Cache(), max_calls=1, backend=backend)
    assert llm.complete([{"role": "user", "content": "a"}]) == "hi"
    assert llm.complete([{"role": "user", "content": "a"}]) == "hi"  # cache hit, no second call
    assert len(calls) == 1 and llm.ledger.cached == 1
    try:
        llm.complete([{"role": "user", "content": "b"}])
        raise AssertionError("expected BudgetExceeded")
    except BudgetExceeded:
        pass


def test_reliability_warning():
    assert reliability_warning(np.zeros(40)) is not None
    assert reliability_warning(np.array([1.0, -1.0] * 20)) is None
