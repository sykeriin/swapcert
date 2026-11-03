# swapcert

Decide whether swapping the model behind an agent (model A → B) lowers task success, using many
cheap **simulated** rollouts plus a few **real** ones, with a confidence interval on
Δsuccess = success(B) − success(A) that stays valid however bad the simulator is.

Status: **everything buildable without the laptop is built and tested against mocks/toy data** (statistics, forking, worldbook, LLM tool simulator, judge, novelty, tau2 adapter, `bench-local`, `record`, toy experiment + figures). **Nothing has run against a real tau2 or a real LLM yet.** The tau2 driver for B's real agent loop is not built. The statistics core, trace schema, forking, a toy
environment and the CLI work and are tested. The τ²-bench integration and the LLM-backed
simulator (plan weeks 1–4) are **not built yet**. Nothing here is evidence about real agents.

## What exists
| Part | Where |
|---|---|
| Real-only, PPI++ (cross-fit; finite-population or superpopulation target), active/IPW PPI, betting CI, ship / no-ship / inconclusive rule | `swapcert/stats/` |
| Trajectory schema + OpenTelemetry GenAI import/export | `swapcert/traces/` |
| Exact-prefix forking (replay A until B first acts differently, then simulate) | `swapcert/fork/prefix.py` |
| Cache (sqlite), LiteLLM wrapper with call/token ledger and `max_calls` cap | `swapcert/llm/` |
| Toy stateful env, mock agents, mock trace-built simulator | `swapcert/envs/toy.py`, `swapcert/sim/toy_sim.py` |
| Offline resampling harness (coverage, width, decision accuracy vs real budget) | `swapcert/exp/resample.py` |

## Use
```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/pytest                      # includes Monte Carlo coverage tests (~30 s)
.venv/bin/swapcert demo               # toy end-to-end experiment
.venv/bin/swapcert simulate t.csv --toy --n-real 60   # fork B on A's runs -> per-task table
.venv/bin/swapcert experiment out/ --toy --active     # 4 toy swap pairs -> tables + SVG figures
.venv/bin/swapcert record A --tasks 20 --trials 2     # (laptop) resumable tau2 run for one model
.venv/bin/swapcert estimate t.csv --tolerance 0.05     # table -> decision report
.venv/bin/swapcert estimate table.csv --tolerance 0.10   # task_id,y_a,s_b,y_b(blank=not run),novelty
```

## Laptop runbook (WSL, RTX 3050 4 GB)
```bash
# terminal 1: agent model on the GPU
ollama serve                                   # :11434
ollama pull qwen3:4b
# terminal 2: FIXED user simulator, CPU only (skip for telecom solo mode)
CUDA_VISIBLE_DEVICES="" OLLAMA_HOST=127.0.0.1:11435 ollama serve
OLLAMA_HOST=127.0.0.1:11435 ollama pull qwen3:1.7b
# terminal 3: install tau2-bench next to this repo (needs Python 3.12-3.13), then
uv run swapcert bench-local --config exp/laptop.yaml --tasks 10   # prints speed + gates G1-G3
```
Edit `exp/laptop.yaml` (`tau2_cmd`, `out_dir`) to point at your tau2 install. See `docs/tau2-notes.md`
for what is verified and what is not.

## What to do next
See [`docs/ROADMAP.md`](docs/ROADMAP.md): status table, the laptop step that unblocks everything, a
decision tree on the `bench-local` gates, build order, timeline and the list of unvalidated assumptions.

## Known gaps in the simulator pipeline
- After the fork, an agent message addressed to the user ends the episode unless the simulator has
  a `reply` method; there is no user-simulator arm yet, so tasks needing user interaction are
  judged on a truncated transcript.
- Novelty uses (tool, previous tool) only, not arguments.
- Tests use scripted mock LLMs: they prove plumbing and invariants (no model leakage, determinism,
  counted fallbacks), not simulator fidelity.

## Honest limitations (read before trusting any number)
- **Validity needs the real runs to be sampled uniformly** (or with logged inclusion probabilities).
  Simulator quality affects width, not validity.
- **Normal-approximation intervals undercover when few real runs changed outcome.** In our Monte
  Carlo, ~85–90% coverage at ~2–5 discordant runs. The tool warns below 15. The betting interval is
  valid at any n but ~3–4× wider, so it is rarely decisive at n ≲ 100.
- **Choose the target deliberately:** `finite_population=True` = these exact tasks (benchmark
  ground truth); `False` = future traffic.
- **Still needs real runs of B.** It shrinks a live A/B test; it does not remove it.
- **Compare against the cheap baseline.** In the toy, using only A's logged outcome as the proxy
  (P1) was as good as the simulator. A simulator has to beat P1 to be worth its cost.
- **Active (novelty-weighted) sampling was worse than uniform** on synthetic data (wider, lower
  coverage). It is not part of any claim until it wins on real data.
- The toy results show the plumbing works; they say nothing about LLM agents.

## Related work
The Replay Gap (arXiv 2608.08239), Trace2Env (2610.06100), SureSim (2510.04354), GLIDE (2605.31278),
PPI (Angelopoulos et al., 2023), PPI++, cross-PPI, active inference (Zrnic & Candès), betting CIs
(Waudby-Smith & Ramdas). Full plan: see the project plan file.
