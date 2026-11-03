import json
import sys
import xml.etree.ElementTree as ET

import numpy as np
from typer.testing import CliRunner

from swapcert import bench
from swapcert.cli import app
from swapcert.exp import resample as R
from swapcert.exp.figures import line_chart_svg
from swapcert.exp.runner import run_experiment
from swapcert.serving import Endpoint, ServingConfig
from swapcert.stats.synthetic import make_population
from tests.test_bench import FIX, STUB


def test_svg_chart_is_valid_xml_and_escapes(tmp_path):
    p = tmp_path / "c.svg"
    line_chart_svg({"a <b>": [(30, 0.1), (60, 0.5)], "c": [(30, 0.2), (60, 0.9)]}, "T & title", "x", "y", p,
                   ylim=(0, 1), hline=(0.8, "target"))
    root = ET.parse(p).getroot()
    assert root.tag.endswith("svg") and len(root.findall(".//{*}polyline")) == 2
    assert "a &lt;b&gt;" in p.read_text()


def test_svg_chart_handles_empty_and_single_point(tmp_path):
    line_chart_svg({}, "empty", "x", "y", tmp_path / "e.svg")
    line_chart_svg({"s": [(5, 0.5)]}, "one", "x", "y", tmp_path / "o.svg")
    ET.parse(tmp_path / "e.svg")
    ET.parse(tmp_path / "o.svg")


def test_active_option_adds_method_and_stays_valid():
    rng = np.random.default_rng(0)
    pop = make_population(200, delta=-0.1, sim_quality=0.8, rng=rng)
    rows = R.run(pop.y_a, pop.y_b, pop.s_b, pop.novelty, [60], 0.05, reps=120, active=True)
    names = {r.method for r in rows}
    assert "P6 active" in names
    act = next(r for r in rows if r.method == "P6 active")
    assert act.coverage >= 0.85 and 0 < act.width < 1  # loose: 120 reps; real check is the slow MC test


def test_run_experiment_toy_outputs_and_decisions(tmp_path):
    res = run_experiment(tmp_path, n_tasks=150, reps=40, budgets=(30, 60))
    for f in ("results.json", "summary.md", "fig_correct.svg", "fig_width.svg"):
        assert (tmp_path / f).exists()
    ET.parse(tmp_path / "fig_correct.svg")
    pairs = res["pairs"]
    assert abs(pairs["A->A control"]["true_delta"]) < 0.12  # noise only; not exactly 0 by construction
    big = pairs["large downgrade"]
    assert big["true_verdict"] == "NO_SHIP" and not big["borderline"]
    assert big["single_draw"]["verdict"] == "NO_SHIP" and big["single_draw"]["covers_truth"]
    assert json.loads((tmp_path / "results.json").read_text())["tolerance"] == 0.05
    assert "no claim about real LLM agents" in (tmp_path / "summary.md").read_text()


def test_record_wraps_tau2_resumably(tmp_path, monkeypatch):
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    monkeypatch.setenv("STUB_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("STUB_FIXTURE", str(FIX))
    cfg = ServingConfig(agent=Endpoint("ollama_chat/qwen3:4b"), tau2_cmd=[sys.executable, str(stub)],
                        out_dir=str(tmp_path / "out"))
    p = bench.record(cfg, ["0", "1"], "A", num_trials=2, agent=Endpoint("ollama_chat/qwen3:1.7b"))
    assert p.exists()
    args = (tmp_path / "out" / "args.txt").read_text().split("\n")
    assert "--auto-resume" in args and args[args.index("--agent-llm") + 1] == "ollama_chat/qwen3:1.7b"
    assert args[args.index("--num-trials") + 1] == "2"


def test_record_and_experiment_cli(tmp_path, monkeypatch):
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    monkeypatch.setenv("STUB_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("STUB_FIXTURE", str(FIX))
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text(f"agent: {{model: ollama_chat/qwen3:4b}}\ntau2_cmd: ['{sys.executable}', '{stub}']\n"
                    f"out_dir: {tmp_path / 'out'}\n")
    r = CliRunner()
    res = r.invoke(app, ["record", "A", "--config", str(cfgp), "--tasks", "2", "--out", str(tmp_path / "A.jsonl")])
    assert res.exit_code == 0, res.output
    assert "2 runs" in res.output and (tmp_path / "A.jsonl").exists()
    res = r.invoke(app, ["experiment", str(tmp_path / "e"), "--toy", "--tasks", "100", "--reps", "20"])
    assert res.exit_code == 0, res.output
    assert "large downgrade" in res.output and (tmp_path / "e" / "fig_correct.svg").exists()
    assert r.invoke(app, ["experiment", str(tmp_path / "x")]).exit_code != 0  # non-toy refused honestly


def test_cross_fit_with_empty_unlabelled_set():
    from swapcert.stats.estimators import cross_fit
    x = np.random.default_rng(0).normal(size=(30, 2))
    f_r, f_u = cross_fit(x, x[:, 0], np.empty((0, 2)))
    assert len(f_r) == 30 and len(f_u) == 0


def test_run_experiment_drops_budgets_covering_all_tasks(tmp_path):
    import pytest
    res = run_experiment(tmp_path, n_tasks=80, reps=10, budgets=(30, 80, 120))
    assert {r["n"] for r in next(iter(res["pairs"].values()))["rows"]} == {30}
    with pytest.raises(ValueError, match="all budgets"):
        run_experiment(tmp_path / "x", n_tasks=20, reps=5, budgets=(30,))
