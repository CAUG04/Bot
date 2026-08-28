"""Métricas de un backtest: winrate con intervalo de confianza, EV, drawdown,
y desgloses por hora del día y por expiración (para responder "cuándo entrar
y con qué tiempo").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pobot.edge import breakeven_winrate, has_demonstrated_edge, wilson_interval
from pobot.types import Trade, TradeResult


@dataclass
class SegmentMetrics:
    label: str
    n_trades: int
    wins: int
    losses: int
    ties: int
    winrate: float
    wilson_lower: float
    wilson_upper: float
    breakeven_winrate: float
    has_edge: bool
    total_pnl: float
    ev_per_trade: float


def _segment_metrics(label: str, trades: list[Trade], payout: float) -> SegmentMetrics:
    wins = sum(1 for t in trades if t.result is TradeResult.WIN)
    losses = sum(1 for t in trades if t.result is TradeResult.LOSS)
    ties = sum(1 for t in trades if t.result is TradeResult.TIE)
    decisive = wins + losses  # los empates no cuentan para el winrate (se devuelve el stake)
    winrate = wins / decisive if decisive > 0 else 0.0
    lower, upper = wilson_interval(wins, decisive) if decisive > 0 else (0.0, 0.0)
    be = breakeven_winrate(payout)
    total_pnl = sum(t.pnl for t in trades)
    ev_per_trade = total_pnl / len(trades) if trades else 0.0
    return SegmentMetrics(
        label=label,
        n_trades=len(trades),
        wins=wins,
        losses=losses,
        ties=ties,
        winrate=winrate,
        wilson_lower=lower,
        wilson_upper=upper,
        breakeven_winrate=be,
        has_edge=has_demonstrated_edge(wins, decisive, payout) if decisive > 0 else False,
        total_pnl=total_pnl,
        ev_per_trade=ev_per_trade,
    )


@dataclass
class DrawdownStats:
    max_drawdown: float
    max_drawdown_pct: float
    longest_losing_streak: int
    longest_winning_streak: int


def _drawdown_stats(trades: list[Trade]) -> DrawdownStats:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    max_dd_pct = 0.0
    cur_win_streak = 0
    cur_loss_streak = 0
    max_win_streak = 0
    max_loss_streak = 0

    for t in trades:
        equity += t.pnl
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)
        if peak > 0:
            max_dd_pct = max(max_dd_pct, dd / peak)

        if t.result is TradeResult.WIN:
            cur_win_streak += 1
            cur_loss_streak = 0
        elif t.result is TradeResult.LOSS:
            cur_loss_streak += 1
            cur_win_streak = 0
        else:
            cur_win_streak = 0
            cur_loss_streak = 0
        max_win_streak = max(max_win_streak, cur_win_streak)
        max_loss_streak = max(max_loss_streak, cur_loss_streak)

    return DrawdownStats(
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        longest_losing_streak=max_loss_streak,
        longest_winning_streak=max_win_streak,
    )


@dataclass
class FullReport:
    overall: SegmentMetrics
    by_hour: dict[int, SegmentMetrics]
    by_expiry: dict[int, SegmentMetrics]
    drawdown: DrawdownStats
    profit_factor: float

    def summary_lines(self) -> list[str]:
        o = self.overall
        edge_str = "SÍ" if o.has_edge else "NO"
        lines = [
            f"Operaciones: {o.n_trades} (wins={o.wins}, losses={o.losses}, ties={o.ties})",
            f"Winrate observado: {o.winrate:.2%}  (IC 95% Wilson: [{o.wilson_lower:.2%}, {o.wilson_upper:.2%}])",
            f"Winrate de equilibrio (payout considerado): {o.breakeven_winrate:.2%}",
            f"¿Ventaja estadísticamente demostrada?: {edge_str}",
            f"EV por operación: {o.ev_per_trade:.4f}   PnL total: {o.total_pnl:.4f}",
            f"Profit factor: {self.profit_factor:.3f}",
            f"Máximo drawdown: {self.drawdown.max_drawdown:.4f} ({self.drawdown.max_drawdown_pct:.2%})",
            f"Racha máx. de pérdidas: {self.drawdown.longest_losing_streak}   "
            f"Racha máx. de ganancias: {self.drawdown.longest_winning_streak}",
        ]
        return lines


def _profit_factor(trades: list[Trade]) -> float:
    gains = sum(t.pnl for t in trades if t.pnl > 0)
    losses = sum(-t.pnl for t in trades if t.pnl < 0)
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def compute_report(trades: list[Trade], payout: float) -> FullReport:
    """Calcula el reporte completo. `payout` es el que se usa para el
    winrate de equilibrio del segmento global; cada trade ya trae su propio
    payout registrado para el cálculo de PnL."""
    overall = _segment_metrics("overall", trades, payout)

    by_hour: dict[int, list[Trade]] = {}
    by_expiry: dict[int, list[Trade]] = {}
    for t in trades:
        hour = (t.entry_timestamp // 3600) % 24
        by_hour.setdefault(hour, []).append(t)
        by_expiry.setdefault(t.expiry_bars, []).append(t)

    hour_metrics = {h: _segment_metrics(f"hour={h}", ts, payout) for h, ts in sorted(by_hour.items())}
    expiry_metrics = {e: _segment_metrics(f"expiry={e}", ts, payout) for e, ts in sorted(by_expiry.items())}

    return FullReport(
        overall=overall,
        by_hour=hour_metrics,
        by_expiry=expiry_metrics,
        drawdown=_drawdown_stats(trades),
        profit_factor=_profit_factor(trades),
    )
