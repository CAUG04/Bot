"""Formato de texto legible para un `FullReport` de backtest/walk-forward."""

from __future__ import annotations

from pobot.backtest.metrics import FullReport


def format_full_report(report: FullReport, title: str = "") -> str:
    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
    lines.extend(report.summary_lines())

    if report.by_hour:
        lines.append("")
        lines.append("Desglose por hora de entrada (UTC):")
        for hour, seg in sorted(report.by_hour.items()):
            edge = "ventaja" if seg.has_edge else "sin ventaja"
            lines.append(
                f"  {hour:02d}h  n={seg.n_trades:4d}  winrate={seg.winrate:6.1%}  "
                f"IC95=[{seg.wilson_lower:5.1%},{seg.wilson_upper:5.1%}]  {edge}"
            )

    if report.by_expiry:
        lines.append("")
        lines.append("Desglose por expiración (nº de velas):")
        for expiry, seg in sorted(report.by_expiry.items()):
            edge = "ventaja" if seg.has_edge else "sin ventaja"
            lines.append(
                f"  {expiry:2d} vela(s)  n={seg.n_trades:4d}  winrate={seg.winrate:6.1%}  "
                f"IC95=[{seg.wilson_lower:5.1%},{seg.wilson_upper:5.1%}]  {edge}"
            )

    return "\n".join(lines)
