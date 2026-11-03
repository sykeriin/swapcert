"""swapcert CLI.

  swapcert estimate data.csv --tolerance 0.10     decision from per-task table
  swapcert demo                                   end-to-end toy experiment (no LLM needed)
  swapcert convert-otel in.jsonl out.jsonl        trajectories -> OTel spans
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import typer

from swapcert.report.render import render_markdown
from swapcert.stats import estimators as E
from swapcert.stats.decision import decide, reliability_warning

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def estimate(
    table: Path = typer.Argument(..., help="CSV: task_id,y_a,s_b[,y_b][,novelty]. Blank y_b = not run for real."),
    tolerance: float = typer.Option(0.10, help="Largest success-rate drop you accept (δ)."),
    alpha: float = typer.Option(0.05),
    out: Path | None = typer.Option(None, help="Write the Markdown report here."),
):
    rows = list(csv.DictReader(table.open()))
    y_a = np.array([float(r["y_a"]) for r in rows])
    s_b = np.array([float(r["s_b"]) for r in rows])
    nov = np.array([float(r.get("novelty") or 0) for r in rows])
    lab = np.array([bool((r.get("y_b") or "").strip()) for r in rows])
    if lab.sum() < 5:
        raise typer.BadParameter("need at least 5 rows with y_b")
    y_b = np.array([float(r["y_b"]) if l else np.nan for r, l in zip(rows, lab)])
    d = (y_b - y_a)[lab]
    x = np.column_stack([s_b - y_a, y_a] + ([nov] if nov.any() else []))
    f_r, f_u = E.cross_fit(x[lab], d, x[~lab])
    est = E.ppi(d, f_r, f_u, alpha, finite_population=True)
    base = E.classical(d, len(rows), alpha)
    dec = decide(est, tolerance)
    md = render_markdown(table.stem, est, dec, {"real-only baseline": base},
                         float(np.std(d - (est.lam or 0) * f_r, ddof=1)), reliability_warning(d), alpha)
    typer.echo(md)
    if out:
        out.write_text(md)


@app.command()
def demo(tasks: int = 278, tolerance: float = 0.03, reps: int = 300):
    """Toy end-to-end run: mock agents + trace-built mock simulator + forking + resampled estimators."""
    from swapcert.exp import resample as R
    from swapcert.exp.toy_experiment import build_dataset

    D = build_dataset(tasks)
    d = D["y_b"] - D["y_a"]
    typer.echo(f"A success {D['y_a'].mean():.3f} | B success {D['y_b'].mean():.3f} | true Δ {d.mean():+.3f} | "
               f"B never diverged from A on {(D['diverged_at'] == -1).mean():.0%} of tasks")
    rows = R.run(D["y_a"], D["y_b"], D["s_b"], D["nov"], [30, 60, 100], tolerance, reps)
    typer.echo(f"{'method':16s} {'n':>4s} {'cover':>6s} {'width':>6s} {'P(correct)':>10s} {'P(incl)':>8s}")
    for r in rows:
        typer.echo(f"{r.method:16s} {r.n:4d} {r.coverage:6.3f} {r.width:6.3f} {r.p_correct:10.2f} {r.p_inconclusive:8.2f}")


@app.command("bench-local")
def bench_local(
    config: Path = typer.Option(Path("exp/laptop.yaml"), help="Serving config (see exp/laptop.yaml)."),
    tasks: int = typer.Option(10, help="Number of tau2 tasks to run."),
    skip_tau2: bool = typer.Option(False, help="Only measure raw throughput."),
):
    """Measure speed and tau2 success on this machine, then evaluate the go/no-go gates."""
    from swapcert import bench
    from swapcert.serving import load_config

    cfg = load_config(config)
    for ep in [cfg.agent] + ([cfg.user] if cfg.user else []):
        p = bench.probe_endpoint(ep)
        typer.echo(f"{ep.model}: prefill {p.prefill_s:.1f}s for {p.prompt_tokens} tokens, decode {p.decode_tok_s:.1f} tok/s")
    if skip_tau2:
        return
    ids = [str(i) for i in range(tasks)]
    run = bench.run_tau2(cfg, ids)
    gates, _ = bench.evaluate_gates(run, cfg.total_rollouts_planned)
    for g in gates:
        typer.echo(f"[{'PASS' if g.passed else 'FAIL'}] {g.name}: {g.detail}")
    typer.echo("All gates passed." if all(g.passed for g in gates) else "Not all gates passed -- do not scale up yet.")


@app.command("import-tau2")
def import_tau2(results: Path, out: Path, model: str = typer.Option(..., help="Model label for these runs.")):
    """Convert a tau2 results.json (or results dir) into swapcert trajectories (JSONL)."""
    from swapcert.envs.tau2_adapter import load_results
    from swapcert.traces.schema import write_jsonl

    trajs = load_results(results, model)
    write_jsonl(str(out), trajs)
    scored = [t.success for t in trajs if t.success is not None]
    typer.echo(f"{len(trajs)} runs, mean success {sum(scored) / max(len(scored), 1):.3f} -> {out}")


@app.command("convert-otel")
def convert_otel(src: Path, dst: Path):
    from swapcert.traces.otel import to_otel
    from swapcert.traces.schema import read_jsonl

    with dst.open("w") as f:
        for t in read_jsonl(str(src)):
            f.write(json.dumps(to_otel(t)) + "\n")


if __name__ == "__main__":
    app()
