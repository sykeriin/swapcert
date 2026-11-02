"""Markdown decision report."""

from __future__ import annotations

from swapcert.stats.decision import Decision, runs_needed
from swapcert.stats.estimators import Estimate


def render_markdown(title: str, est: Estimate, dec: Decision, extra: dict[str, Estimate] | None = None,
                    resid_sd: float | None = None, warning: str | None = None, alpha: float = 0.05) -> str:
    lines = [
        f"# Model-swap decision: {title}",
        "",
        f"**Verdict: {dec.verdict.value}**  (tolerance δ = {dec.tolerance:.3f}; {dec.reason})",
        "",
        f"- Δ success (B − A): **{est.delta:+.3f}**, {int((1 - alpha) * 100)}% CI [{est.lo:+.3f}, {est.hi:+.3f}]",
        f"- Method: {est.method}" + (f" (λ = {est.lam:.2f})" if est.lam is not None else ""),
        f"- Real runs of B used: {est.n_real} of {est.n_total} tasks",
    ]
    if resid_sd:
        lines.append(f"- Real runs needed for the interval half-width to reach δ/2 at this residual spread: "
                     f"~{runs_needed(resid_sd, dec.tolerance / 2, alpha, est.n_total)}")
    if warning:
        lines += ["", f"> **Warning:** {warning}"]
    if extra:
        lines += ["", "| Method | Δ̂ | CI | width |", "|---|---|---|---|"]
        for name, e in extra.items():
            lines.append(f"| {name} | {e.delta:+.3f} | [{e.lo:+.3f}, {e.hi:+.3f}] | {e.width:.3f} |")
    lines += ["", "_Validity holds for any simulator quality provided the real runs were sampled "
              "uniformly (or with the logged probabilities); simulator quality only changes the width._"]
    return "\n".join(lines) + "\n"
