# τ²-bench integration notes (read from source, 3 Nov 2026)

Pinned reference: `sierra-research/tau2-bench` @ `19f13ea3f5fba8b4c4d272d1353590f29dc53062`.
**Nothing here was run.** tau2 could not be installed in the build container (install of
third-party code was blocked), so every schema statement below comes from reading source.
Treat the adapter as unvalidated until `swapcert import-tau2` has succeeded on a real run.

## Confirmed from source
- Python `>=3.12,<3.14`; depends on `litellm>=1.80.15,<1.82.7`.
- Tasks per domain (base split): airline 50, retail 114, telecom 114 (telecom.json holds 2285 in
  `full`). Total 278. Train/test splits exist for airline (30/20), retail (74/40), telecom (74/40).
- Model calls go through `litellm.completion(model=..., **kwargs)`; `--agent-llm-args` /
  `--user-llm-args` JSON is forwarded as kwargs, so `api_base` should reach an Ollama server.
  **Unverified end-to-end.**
- Result layouts: single `results.json` containing `simulations` (default for text) or a
  directory with `results.json` metadata plus `simulations/*.json`. The per-file names in the
  directory layout were not visible; the adapter globs `*.json`.
- `SimulationRun`: `task_id`, `trial`, `messages`, `reward_info.reward`, `termination_reason`
  (`user_stop|agent_stop|max_steps|timeout|too_many_errors|agent_error|user_error|
  infrastructure_error|context_window_exceeded|unexpected_error`), `duration`, `seed`.
- Messages: `assistant`/`user` (with optional `tool_calls[{id,name,arguments,requestor}]`),
  `tool` (`id`, `content`, `requestor`, `error`), and `MultiToolMessage` (`role: tool`,
  `tool_messages[]`).

## Solo mode (important for the $0 setup)
- `--agent llm_agent_solo --user dummy_user`: no user simulator; the agent works from a *ticket*
  using tool calls only.
- Only **telecom** tasks have tickets (114/114 in base). Airline and retail have 0.
- Upside: no user LLM, no user/agent confound, one model on the GPU, simpler and cheaper.
- Downside: telecom is the hardest domain; a 2–4B model may score ~0%, which fails gate G1.
- `bench-local` should be run for BOTH retail (user simulator on CPU) and telecom (solo), and the
  choice made from the measured success rates.

## Open questions to settle on the laptop
1. Does `api_base` in `--agent-llm-args` reach Ollama (`ollama_chat/<model>`)?
2. Does `ollama_chat/qwen3:4b` return well-formed tool calls (gate G3)?
3. Exact directory-layout filenames; whether `--save-to` writes under tau2's own `data/simulations/`
   (then point `out_dir` there).
4. Qwen3 "thinking" output: may need `/no_think` or `think: false` via llm args to keep latency down.
