"""Validación walk-forward: la única fuente de verdad para decidir si una
estrategia tiene ventaja real. Un backtest in-sample (ajustado y evaluado
sobre los mismos datos) sistemáticamente sobreestima el winrate.

Divide la serie en ventanas rodantes de train/test. Cada ventana de test se
evalúa con la estrategia tal cual (sin volver a ajustar contra ella), y los
trades de todas las ventanas de test se concatenan para el reporte final.
"""

from __future__ import annotations

from dataclasses import dataclass

from pobot.backtest.engine import BacktestConfig, run_backtest
from pobot.backtest.metrics import FullReport, compute_report
from pobot.data.candles import CandleSeries
from pobot.strategies.base import Strategy
from pobot.types import Trade


@dataclass
class WalkForwardWindow:
    train_start: int
    train_end: int  # exclusivo
    test_start: int
    test_end: int  # exclusivo


def make_windows(n: int, train_size: int, test_size: int, step: int | None = None) -> list[WalkForwardWindow]:
    """Genera ventanas rodantes [train][test] que avanzan `step` barras
    (por defecto, `step = test_size`, es decir, sin solape entre test)."""
    if train_size < 1 or test_size < 1:
        raise ValueError("train_size y test_size deben ser >= 1")
    step = step or test_size
    windows = []
    train_start = 0
    while True:
        train_end = train_start + train_size
        test_end = train_end + test_size
        if test_end > n:
            break
        windows.append(WalkForwardWindow(train_start, train_end, train_end, test_end))
        train_start += step
    return windows


def run_walkforward(
    series: CandleSeries,
    strategy_factory,
    config: BacktestConfig,
    train_size: int,
    test_size: int,
    step: int | None = None,
) -> tuple[FullReport, list[Trade], list[WalkForwardWindow]]:
    """`strategy_factory(train_series) -> Strategy` permite que la estrategia
    se "entrene" (ej. ajuste umbrales o un modelo ML) SOLO con la ventana de
    train, y luego se evalúe en la ventana de test correspondiente, que el
    entrenamiento nunca vio.

    Para estrategias sin estado (reglas fijas), `strategy_factory` puede
    ignorar `train_series` y devolver siempre la misma instancia.
    """
    windows = make_windows(len(series), train_size, test_size, step)
    all_trades: list[Trade] = []

    for w in windows:
        train_series = series.slice(w.train_start, w.train_end)
        strategy = strategy_factory(train_series)

        # La ventana de test se evalúa con contexto histórico (incluye train)
        # para que los indicadores tengan warmup, pero solo se aceptan señales
        # y trades cuya barra de señal caiga dentro de [test_start, test_end).
        context_series = series.slice(w.train_start, w.test_end)
        offset = w.train_start
        report = run_backtest(context_series, strategy, config)

        test_start_local = w.test_start - offset
        test_end_local = w.test_end - offset
        for t in report.trades:
            if test_start_local <= t.entry_index - 1 < test_end_local:
                all_trades.append(t)

    full_report = compute_report(all_trades, config.payout)
    return full_report, all_trades, windows
