"""Dependency-free SVG line charts (no matplotlib needed)."""

from __future__ import annotations

from html import escape
from pathlib import Path

# colour-blind-safe categorical palette (Okabe-Ito)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#999999"]


def line_chart_svg(series: dict[str, list[tuple[float, float]]], title: str, xlabel: str, ylabel: str,
                   path: str | Path, ylim: tuple[float, float] | None = None, hline: tuple[float, str] | None = None) -> None:
    W, H, L, R, T, B = 760, 440, 70, 190, 50, 60
    xs = [x for pts in series.values() for x, _ in pts] or [0, 1]
    ys = [y for pts in series.values() for _, y in pts] or [0, 1]
    x0, x1 = min(xs), max(xs)
    y0, y1 = ylim if ylim else (min(0.0, min(ys)), max(ys) * 1.08 or 1.0)
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y1 = y0 + 1

    def px(x: float) -> float:
        return L + (x - x0) / (x1 - x0) * (W - L - R)

    def py(y: float) -> float:
        return H - B - (y - y0) / (y1 - y0) * (H - T - B)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{L}" y="28" font-size="15" font-weight="bold">{escape(title)}</text>',
         f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="#333"/>',
         f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H - B}" stroke="#333"/>']
    for i in range(6):
        yv = y0 + (y1 - y0) * i / 5
        o.append(f'<line x1="{L}" y1="{py(yv):.1f}" x2="{W - R}" y2="{py(yv):.1f}" stroke="#e5e5e5"/>')
        o.append(f'<text x="{L - 8}" y="{py(yv) + 4:.1f}" text-anchor="end">{yv:.2f}</text>')
    for xv in sorted(set(xs)):
        o.append(f'<text x="{px(xv):.1f}" y="{H - B + 18}" text-anchor="middle">{xv:g}</text>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 15}" text-anchor="middle">{escape(xlabel)}</text>')
    o.append(f'<text x="18" y="{(T + H - B) / 2}" text-anchor="middle" transform="rotate(-90 18 {(T + H - B) / 2})">{escape(ylabel)}</text>')
    if hline:
        o.append(f'<line x1="{L}" y1="{py(hline[0]):.1f}" x2="{W - R}" y2="{py(hline[0]):.1f}" stroke="#666" stroke-dasharray="5 4"/>')
        o.append(f'<text x="{W - R + 8}" y="{py(hline[0]) + 4:.1f}" fill="#666">{escape(hline[1])}</text>')
    for k, (name, pts) in enumerate(series.items()):
        c = PALETTE[k % len(PALETTE)]
        d = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in sorted(pts))
        o.append(f'<polyline points="{d}" fill="none" stroke="{c}" stroke-width="2"/>')
        o += [f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" fill="{c}"/>' for x, y in pts]
        ly = T + 8 + k * 20
        o.append(f'<line x1="{W - R + 8}" y1="{ly}" x2="{W - R + 28}" y2="{ly}" stroke="{c}" stroke-width="2"/>')
        o.append(f'<text x="{W - R + 34}" y="{ly + 4}">{escape(name)}</text>')
    o.append("</svg>")
    Path(path).write_text("\n".join(o))
