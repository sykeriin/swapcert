"""`swapcert bench-local`: decide whether this machine can produce usable ground truth.

Measures (1) raw tokens/s and prefill latency of each served model, (2) a small τ² run:
success rate, error-termination rate, minutes per rollout, and projects total GPU-hours for the
planned scale.  Then evaluates the three gates from the plan.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from swapcert.envs.tau2_adapter import load_results, terminal_error_rate
from swapcert.serving import Endpoint, ServingConfig, build_run_args, results_path

G1_MIN_SUCCESS = 0.15
G2_MAX_HOURS = 250.0
G3_MAX_ERROR_RATE = 0.30


@dataclass
class Probe:
    model: str
    prefill_s: float  # time to answer a ~long prompt with 1 token  (≈ prefill latency)
    decode_tok_s: float  # generation speed
    prompt_tokens: int


def _chat(ep: Endpoint, content: str, max_tokens: int, timeout: float = 600) -> tuple[dict, float]:
    body = json.dumps({"model": ep.bare_model, "max_tokens": max_tokens, "temperature": 0,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(ep.openai_base + "/chat/completions", body,
                                 {"Content-Type": "application/json", "Authorization": "Bearer local"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, time.perf_counter() - t0


def probe_endpoint(ep: Endpoint, prompt_words: int = 4500, gen_tokens: int = 128) -> Probe:
    """Prefill latency from a 1-token reply to a long prompt; decode speed from a long generation."""
    long_prompt = ("The customer policy states that orders may be cancelled while pending. " * (prompt_words // 11 + 1))
    long_prompt = " ".join(long_prompt.split()[:prompt_words]) + "\nReply with the single word OK."
    out1, t_prefill = _chat(ep, long_prompt, 1)
    out2, t_gen = _chat(ep, "Count upward from 1 in words, one number per line, as far as you can.", gen_tokens)
    n_out = out2.get("usage", {}).get("completion_tokens", gen_tokens)
    return Probe(ep.model, t_prefill, n_out / max(t_gen, 1e-6), out1.get("usage", {}).get("prompt_tokens", 0))


@dataclass
class TauRun:
    n: int
    success_rate: float
    error_rate: float
    minutes_per_rollout: float
    termination: dict[str, int] = field(default_factory=dict)


def run_tau2(cfg: ServingConfig, task_ids: list[str], save_to: str = "bench") -> TauRun:
    """Run tau2 on a few tasks (resumable) and summarise.  Raises if the subprocess fails."""
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    args = build_run_args(cfg, task_ids, save_to)
    t0 = time.perf_counter()
    subprocess.run(args, check=True)
    wall = time.perf_counter() - t0
    trajs = load_results(results_path(cfg, save_to), cfg.agent.model)
    term: dict[str, int] = {}
    for t in trajs:
        k = str(t.meta.get("termination_reason"))
        term[k] = term.get(k, 0) + 1
    scored = [t.success for t in trajs if t.success is not None]
    return TauRun(len(trajs), sum(scored) / len(scored) if scored else 0.0,
                  terminal_error_rate(trajs), wall / 60 / max(len(trajs), 1), term)


@dataclass
class Gate:
    name: str
    passed: bool
    detail: str


def evaluate_gates(run: TauRun, total_rollouts: int) -> tuple[list[Gate], float]:
    hours = run.minutes_per_rollout * total_rollouts / 60
    gates = [
        Gate("G1 success >= 15%", run.success_rate >= G1_MIN_SUCCESS,
             f"{run.success_rate:.0%} of {run.n} runs (too few runs? rerun on >=30 if borderline)"),
        Gate("G2 projected hours <= 250", hours <= G2_MAX_HOURS,
             f"{run.minutes_per_rollout:.1f} min/rollout x {total_rollouts} = {hours:.0f} h"),
        Gate("G3 error terminations < 30%", run.error_rate < G3_MAX_ERROR_RATE,
             f"{run.error_rate:.0%}; terminations: {run.termination}"),
    ]
    return gates, hours


def record(cfg: ServingConfig, task_ids: list[str], save_to: str, num_trials: int = 1,
           agent: Endpoint | None = None) -> Path:
    """Run `tau2 run` for one agent over the given tasks (resumable via --auto-resume) and return
    the results path.  This is how model A's 'production logs' and each B's ground truth are made."""
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(build_run_args(cfg, task_ids, save_to, num_trials, agent), check=True)
    return results_path(cfg, save_to)
