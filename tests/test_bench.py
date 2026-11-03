import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from swapcert import bench
from swapcert.serving import Endpoint, ServingConfig, build_run_args, load_config

FIX = Path(__file__).parent / "fixtures" / "tau2_results_sample.json"

STUB = """
import os, sys, shutil, pathlib
args = sys.argv[1:]
save_to = args[args.index("--save-to") + 1]
out = pathlib.Path(os.environ["STUB_OUT"]) / save_to
out.mkdir(parents=True, exist_ok=True)
shutil.copy(os.environ["STUB_FIXTURE"], out / "results.json")
open(os.environ["STUB_OUT"] + "/args.txt", "w").write("\\n".join(args))
"""


class FakeOpenAI(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers["Content-Length"])
        req = json.loads(self.rfile.read(n))
        mt = req["max_tokens"]
        time.sleep(0.05 if mt == 1 else 0.2)
        body = json.dumps({"choices": [{"message": {"content": "OK"}}],
                           "usage": {"prompt_tokens": 6000, "completion_tokens": mt}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_probe_measures_prefill_and_decode(server):
    p = bench.probe_endpoint(Endpoint("ollama_chat/qwen3:4b", server), prompt_words=200, gen_tokens=100)
    assert p.prompt_tokens == 6000
    assert 0.04 < p.prefill_s < 1.0
    assert 100 < p.decode_tok_s < 1000  # 100 tokens in ~0.2 s


def test_build_run_args_user_and_solo_modes():
    ag, us = Endpoint("ollama_chat/qwen3:4b"), Endpoint("ollama_chat/qwen3:1.7b", "http://localhost:11435")
    a = build_run_args(ServingConfig(agent=ag, user=us), ["1", "2"], "x")
    assert a[a.index("--user-llm") + 1] == "ollama_chat/qwen3:1.7b"
    assert json.loads(a[a.index("--user-llm-args") + 1])["api_base"] == "http://localhost:11435"
    assert a[a.index("--task-ids"):] == ["--task-ids", "1", "2"] and "--auto-resume" in a
    s = build_run_args(ServingConfig(agent=ag, domain="telecom"), [], "y")
    assert "llm_agent_solo" in s and "dummy_user" in s and "--user-llm" not in s


def test_load_example_config():
    cfg = load_config(Path(__file__).parent.parent / "exp" / "laptop.yaml")
    assert cfg.agent.model.startswith("ollama_chat/") and cfg.user is not None and not cfg.solo


def test_run_tau2_with_stub_and_gates(tmp_path, monkeypatch):
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    monkeypatch.setenv("STUB_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("STUB_FIXTURE", str(FIX))
    cfg = ServingConfig(agent=Endpoint("ollama_chat/qwen3:4b"), tau2_cmd=[sys.executable, str(stub)],
                        out_dir=str(tmp_path / "out"))
    run = bench.run_tau2(cfg, ["0", "1"])
    assert run.n == 2 and run.success_rate == 0.5 and run.error_rate == 0.5
    assert "--auto-resume" in (tmp_path / "out" / "args.txt").read_text()
    gates, hours = bench.evaluate_gates(run, 4500)
    assert [g.passed for g in gates] == [True, True, False]  # error rate 50% fails G3
    assert hours >= 0


def test_gate_logic_boundaries():
    ok = bench.TauRun(10, 0.15, 0.29, 3.0)
    gates, hours = bench.evaluate_gates(ok, 4500)
    assert all(g.passed for g in gates) and hours == pytest.approx(225)
    slow = bench.TauRun(10, 0.5, 0.0, 4.0)  # 300 h
    assert not bench.evaluate_gates(slow, 4500)[0][1].passed
    low = bench.TauRun(10, 0.1, 0.0, 1.0)
    assert not bench.evaluate_gates(low, 4500)[0][0].passed
