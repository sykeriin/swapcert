import threading
from http.server import HTTPServer

from typer.testing import CliRunner

from swapcert.cli import app
from swapcert.traces.schema import read_jsonl
from tests.test_bench import FIX, FakeOpenAI

runner = CliRunner()


def test_import_tau2_command(tmp_path):
    out = tmp_path / "A.jsonl"
    r = runner.invoke(app, ["import-tau2", str(FIX), str(out), "--model", "ollama_chat/qwen3:4b"])
    assert r.exit_code == 0, r.output
    assert "2 runs, mean success 0.500" in r.output
    assert [t.task_id for t in read_jsonl(str(out))] == ["0", "1"]


def test_bench_local_throughput_only(tmp_path):
    srv = HTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"agent: {{model: ollama_chat/qwen3:4b, api_base: '{base}'}}\n"
                   f"user: {{model: ollama_chat/qwen3:1.7b, api_base: '{base}'}}\n")
    r = runner.invoke(app, ["bench-local", "--config", str(cfg), "--skip-tau2"])
    srv.shutdown()
    assert r.exit_code == 0, r.output
    assert "qwen3:4b" in r.output and "tok/s" in r.output and "qwen3:1.7b" in r.output


def test_estimate_command_end_to_end(tmp_path):
    import csv

    import numpy as np
    rng = np.random.default_rng(0)
    p = tmp_path / "t.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "y_a", "s_b", "y_b", "novelty"])
        for i in range(120):
            ya = float(rng.random() < 0.6)
            yb = float(rng.random() < (0.85 if ya else 0.3))
            w.writerow([i, ya, yb if rng.random() < 0.8 else 1 - yb, yb if i < 40 else "", rng.random()])
    r = runner.invoke(app, ["estimate", str(p), "--tolerance", "0.1"])
    assert r.exit_code == 0, r.output
    assert "Verdict:" in r.output and "95% CI" in r.output
