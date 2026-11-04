# Roadmap and next steps (written 3 Nov 2026)

## 1. Where things stand

| Piece | Built | Tested | Validated on real data |
|---|---|---|---|
| Statistics: real-only, PPI++ (cross-fit), active/IPW PPI, betting CI, decision rule | yes | Monte Carlo coverage tests (synthetic) | no |
| Trace schema, OTel import/export, sqlite LLM cache, call/token ledger | yes | unit tests | no |
| Forking with text-aware matching, worldbook, LLM tool simulator, outcome judge, novelty | yes | scripted mock LLMs only | **no** |
| τ² adapter, serving config, `bench-local`, `record`, `import-tau2` | yes | stub `tau2` command + hand-written fixture | **no** (schema read from source, never run) |
| Toy multi-pair experiment, figures (`swapcert experiment --toy`) | yes | tests | n/a (toy) |
| τ² driver that runs B's real agent loop on a forked history | **no** | - | - |
| User-simulator arm, per-step router agent, prompt tuning for 4B models | **no** | - | - |

Nothing in this repo is evidence about real LLM agents yet. The toy results only show that the
plumbing and the statistics behave.

## 2. Your next action (needs the laptop; ~1 hour)

Follow the README "Laptop runbook", then run, once per option, and send the full output:

```bash
# Option U: retail with a CPU-only user simulator (keep the `user:` block in exp/laptop.yaml)
uv run swapcert bench-local --config exp/laptop.yaml --tasks 10

# Option S: telecom in solo mode (set domain: telecom and delete the `user:` block)
uv run swapcert bench-local --config exp/laptop.yaml --tasks 10
```

Also send: `nvidia-smi` inside WSL while a model is loaded (to confirm the GPU is used), and the first
error if anything fails to import or parse.

## 3. Decision tree on the result

| Outcome | What it means | What happens next |
|---|---|---|
| Adapter or tau2 call fails | assumed schema / `api_base` forwarding is wrong | fix `tau2_adapter.py` and the fixture from the real file; re-run |
| G1 fails (success < 15%) in both options | 2-4B models are too weak for tau2 here | try qwen3:8b with CPU offload on airline (50 tasks) only; or switch benchmark; or report that the setting is infeasible |
| G1 passes for U only | retail/airline with a CPU user simulator is the path | build the user-simulator arm |
| G1 passes for S only | telecom solo is the path (114 tasks, no user confound) | skip the user-simulator arm entirely |
| G2 fails (>250 h projected) | too slow | cut pairs from 5 to 3 and/or trials from 2 to 1; shrink the task set |
| G3 fails (>30% error terminations) | malformed tool calls from the small model | try another model family (llama3.2:3b) or constrain decoding |
| All pass | green light | section 4 |

## 4. Build order once the gates pass

1. **Real-agent fork driver (the biggest remaining piece).** Run B's actual agent loop (tau2's
   `LLMAgent` with B's model) on a forked message history, then hand the tail to the simulator.
   Needs a tau2 install; mock-tested here, validated on the laptop. Expect rework: it depends on tau2
   internals I could only read, not run.
2. `swapcert record` model A at scale (resumable; 2 trials per task), then each B (ground truth).
3. Real `swapcert simulate` (not just `--toy`), `judge-calibrate` against A's real rewards, and
   simulator-health counters (fallbacks, unknown tools).
4. Measure: divergence-step distribution (does forking help at all?), A->A noise, rho vs the
   cheap A-as-proxy baseline P1.
5. **Go/no-go (original week 4): rho beyond P1 >= 0.4.** If not, the honest result is "simulation does not
   help beyond logged outcomes in this setting" -- write that up; it is still a result.
6. Remaining pairs, resampling experiments, headline figure, write-up.

## 5. Revised timeline (assumes the gate run happens in the week of 9 Nov)

| Window | Goal |
|---|---|
| 9-15 Nov | `bench-local` results; fix adapter; choose U vs S |
| 16-29 Nov | fork driver; record A; first real B (A->A control + one downgrade) |
| 30 Nov-13 Dec | simulator calibration; rho vs P1; go/no-go |
| 14 Dec-10 Jan | remaining pairs overnight; figures; write-up (holidays will slow this) |
| mid Jan 2027 | arXiv preprint at the earliest; a workshop is a more realistic first venue |

GPU-hours (150-250 h total) are the binding constraint at $0, so expect slippage of 1-2 weeks.

## 6. Unvalidated assumptions (check these first when something breaks)

1. tau2 result JSON layout and message fields (read from source commit `19f13ea`, never run).
2. `--agent-llm-args '{"api_base": ...}'` reaches Ollama through LiteLLM (`ollama_chat/<model>`).
3. The directory-layout results filenames (`simulations/*.json` is a guess).
4. Qwen3 "thinking" output may need disabling for latency.
5. A fixed small CPU user simulator gives usable ground truth.
6. Two Ollama servers (GPU :11434, CPU :11435) coexist in WSL within 16 GB RAM.

## 7. Findings so far that should shape the write-up

- Normal-approximation intervals undercover (85-90%) when fewer than ~15 labelled runs changed outcome; `swapcert`
  warns. The betting interval is valid but 3-4x wider, so rarely decisive at n <~ 100.
- On toy data the A-as-proxy baseline (P1) matches the simulator arms, and novelty-weighted active sampling did not beat
  uniform sampling. Both need re-testing on real data; do not claim either way yet.
- Toy A->A control: true delta averaged -0.014 (spread 0.014 across 8 task sets of 300), single draws on 120-200 tasks reached
  -0.05 to -0.10 from sampling noise of one logged run per task. A small negative offset (~2.8 s.e.) is unexplained; it is
  immaterial at a 0.05 tolerance but check the real A->A control for a systematic gap before trusting any pair.
- With tau2's 278 tasks a +/-5pp interval is out of reach; plan on a ~10pp tolerance.
- Position against GLIDE (arXiv 2605.31278), which already applies PPI to agent evaluation with a judge as the proxy.
- Results on 2-4B models say little about swaps between frontier models.
- The method still needs a few real runs of B; it shrinks a live A/B test, it does not remove it.
