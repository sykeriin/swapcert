# swapcert

Decide whether swapping the model behind an agent (model A → B) lowers task success, using many
cheap **simulated** rollouts plus a few **real** ones, with a confidence interval on
Δsuccess = success(B) − success(A) that stays valid however bad the simulator is.

Status: **Phase 0 (no LLM needed).** The statistics core, trace schema, forking, a toy
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
.venv/bin/swapcert estimate table.csv --tolerance 0.10   # task_id,y_a,s_b,y_b(blank=not run),novelty
```

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
