"""Serving configuration (which model answers where) and `tau2 run` command construction.

Design (see plan, Stage 2): the agent model runs on the GPU server; the user simulator is a
FIXED small model on a separate CPU-only server, so swapping the agent never changes the user and
never forces weight reloads on a 4 GB card.  Solo mode (telecom only) needs no user simulator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Endpoint:
    model: str  # LiteLLM name, e.g. "ollama_chat/qwen3:4b"
    api_base: str = "http://localhost:11434"
    temperature: float = 0.0
    extra: dict = field(default_factory=dict)

    def llm_args(self) -> str:
        return json.dumps({"api_base": self.api_base, "temperature": self.temperature, **self.extra})

    @property
    def openai_base(self) -> str:
        """OpenAI-compatible root, used by the throughput probe."""
        return self.api_base.rstrip("/") + "/v1"

    @property
    def bare_model(self) -> str:
        return self.model.split("/", 1)[1] if "/" in self.model else self.model


@dataclass
class ServingConfig:
    agent: Endpoint
    user: Endpoint | None = None  # None => solo mode
    domain: str = "retail"
    max_steps: int = 40
    seed: int = 300
    tau2_cmd: list[str] = field(default_factory=lambda: ["tau2"])
    out_dir: str = "runs/tau2"
    total_rollouts_planned: int = 4500  # plan §5.0 scale; used for the GPU-hour projection

    @property
    def solo(self) -> bool:
        return self.user is None


def load_config(path: str | Path) -> ServingConfig:
    raw = yaml.safe_load(Path(path).read_text())
    user = raw.pop("user", None)
    agent = Endpoint(**raw.pop("agent"))
    return ServingConfig(agent=agent, user=Endpoint(**user) if user else None, **raw)


def build_run_args(cfg: ServingConfig, task_ids: list[str], save_to: str, num_trials: int = 1,
                   agent: Endpoint | None = None) -> list[str]:
    """Argument list for `tau2 run` (flags verified against tau2-bench docs/cli-reference.md)."""
    ag = agent or cfg.agent
    args = [*cfg.tau2_cmd, "run", "--domain", cfg.domain,
            "--agent-llm", ag.model, "--agent-llm-args", ag.llm_args(),
            "--num-trials", str(num_trials), "--max-steps", str(cfg.max_steps),
            "--max-concurrency", "1", "--seed", str(cfg.seed),
            "--save-to", save_to, "--auto-resume"]
    if cfg.solo:
        args += ["--agent", "llm_agent_solo", "--user", "dummy_user"]
    else:
        assert cfg.user is not None
        args += ["--user-llm", cfg.user.model, "--user-llm-args", cfg.user.llm_args()]
    if task_ids:
        args += ["--task-ids", *task_ids]
    return args


def results_path(cfg: ServingConfig, save_to: str) -> Path:
    # tau2 writes data/simulations/<save_to>/results.json relative to ITS data dir; we point the
    # adapter at whichever location the user configures via out_dir.
    return Path(cfg.out_dir) / save_to / "results.json"
