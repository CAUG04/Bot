"""Simulador de opciones binarias barra a barra.

Regla de ejecución (idéntica a `pobot.labeling`): la señal se evalúa al
cierre de la barra `i`, la operación entra al **open de i+1** y se liquida
al **close de i+horizon**. No hay slippage ni comisión explícita: el
`payout` ya es neto.

Filtros soportados (todos validables en el propio backtest, no supuestos):
- `cooldown_bars`: barras mínimas de espera tras abrir una operación.
- `max_trades_per_day`: límite de operaciones por día UTC.
- `allowed_hours`: horas UTC (0-23) en las que se permite operar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pobot.data.candles import CandleSeries
from pobot.strategies.base import Strategy
from pobot.types import Direction, Signal, Trade, TradeResult


@dataclass
class BacktestConfig:
    payout: float = 0.85
    stake: float = 1.0
    cooldown_bars: int = 1
    max_trades_per_day: Optional[int] = None
    allowed_hours: Optional[list[int]] = None
    tie_policy: str = "refund"  # refund | loss


@dataclass
class BacktestReport:
    trades: list[Trade] = field(default_factory=list)
    signals_seen: int = 0
    signals_filtered: int = 0

    @property
    def n_trades(self) -> int:
        return len(self.trades)


def _hour_of_day(timestamp: int) -> int:
    return (timestamp // 3600) % 24


def _day_bucket(timestamp: int) -> int:
    return timestamp // 86400


def run_backtest(series: CandleSeries, strategy: Strategy, config: BacktestConfig) -> BacktestReport:
    """Ejecuta el backtest completo y devuelve el reporte con la lista de trades.

    El bucle avanza barra a barra y nunca le pasa a la estrategia información
    de índices futuros: `strategy.evaluate(series, i)` solo ve hasta `i`.
    """
    report = BacktestReport()
    n = len(series)
    next_available_index = strategy.warmup()
    trades_today: dict[int, int] = {}

    for i in range(strategy.warmup(), n):
        if i < next_available_index:
            continue

        signal = strategy.evaluate(series, i)
        if signal is None:
            continue
        report.signals_seen += 1

        entry_index = i + 1
        exit_index = i + signal.expiry_bars
        if exit_index >= n:
            report.signals_filtered += 1
            continue  # no hay futuro suficiente para liquidar esta operación

        entry_candle = series[entry_index]
        if config.allowed_hours is not None and _hour_of_day(entry_candle.timestamp) not in config.allowed_hours:
            report.signals_filtered += 1
            continue

        if config.max_trades_per_day is not None:
            day = _day_bucket(entry_candle.timestamp)
            if trades_today.get(day, 0) >= config.max_trades_per_day:
                report.signals_filtered += 1
                continue

        exit_candle = series[exit_index]
        trade = Trade(
            entry_index=entry_index,
            entry_timestamp=entry_candle.timestamp,
            exit_index=exit_index,
            exit_timestamp=exit_candle.timestamp,
            direction=signal.direction,
            entry_price=entry_candle.open,
            exit_price=exit_candle.close,
            expiry_bars=signal.expiry_bars,
            stake=config.stake,
            payout=config.payout,
            strategy=signal.strategy,
            confidence=signal.confidence,
        )

        if trade.result is TradeResult.TIE and config.tie_policy == "loss":
            trade.forced_result = TradeResult.LOSS

        report.trades.append(trade)
        if config.max_trades_per_day is not None:
            day = _day_bucket(entry_candle.timestamp)
            trades_today[day] = trades_today.get(day, 0) + 1

        next_available_index = exit_index + 1 + max(0, config.cooldown_bars - 1)

    return report
