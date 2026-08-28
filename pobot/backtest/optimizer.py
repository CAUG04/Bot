"""Búsqueda de parámetros de estrategia, puntuada siempre out-of-sample.

Advertencia de diseño: probar muchas combinaciones de parámetros y quedarse
con la mejor es, en sí mismo, una forma de overfitting (data dredging). Este
optimizador lo mitiga de dos formas:

1. La puntuación de cada combinación viene de `run_walkforward` (out-of-sample
   por construcción), nunca de un ajuste in-sample.
2. Aplica una corrección tipo Bonferroni al umbral de significancia de Wilson:
   con `k` combinaciones probadas, exige que el límite inferior del intervalo
   de Wilson supere el equilibrio usando `z` más exigente (alpha / k), no el
   z=1.96 por defecto. Cuantas más combinaciones se prueban, más difícil es
   que una parezca ganadora por puro azar.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Callable, Iterable

from pobot.backtest.engine import BacktestConfig
from pobot.backtest.metrics import FullReport
from pobot.backtest.walkforward import run_walkforward
from pobot.data.candles import CandleSeries
from pobot.edge import breakeven_winrate, wilson_interval
from pobot.strategies.base import Strategy

# z aproximados para alpha de dos colas / k (Bonferroni), k -> z
_BONFERRONI_Z = {1: 1.96, 5: 2.58, 10: 2.81, 25: 3.02, 50: 3.29, 100: 3.48, 250: 3.72}


def _z_for_trials(k: int) -> float:
    for threshold in sorted(_BONFERRONI_Z):
        if k <= threshold:
            return _BONFERRONI_Z[threshold]
    return 3.9  # umbral muy exigente para búsquedas grandes


@dataclass
class OptimizationResult:
    params: dict
    report: FullReport
    corrected_has_edge: bool
    z_used: float
    rank_score: float  # límite inferior de Wilson con la z corregida (para ordenar)


def grid_search(
    series: CandleSeries,
    param_grid: dict[str, list],
    strategy_builder: Callable[[dict], Callable[[CandleSeries], Strategy]],
    backtest_config: BacktestConfig,
    train_size: int,
    test_size: int,
    step: int | None = None,
) -> list[OptimizationResult]:
    """Prueba todas las combinaciones de `param_grid` (producto cartesiano)."""
    keys = list(param_grid.keys())
    combos = [dict(zip(keys, values)) for values in itertools.product(*param_grid.values())]
    return _evaluate_combos(combos, series, strategy_builder, backtest_config, train_size, test_size, step)


def random_search(
    series: CandleSeries,
    param_space: dict[str, list],
    n_trials: int,
    strategy_builder: Callable[[dict], Callable[[CandleSeries], Strategy]],
    backtest_config: BacktestConfig,
    train_size: int,
    test_size: int,
    step: int | None = None,
    seed: int | None = 0,
) -> list[OptimizationResult]:
    rng = random.Random(seed)
    keys = list(param_space.keys())
    combos = [{k: rng.choice(param_space[k]) for k in keys} for _ in range(n_trials)]
    return _evaluate_combos(combos, series, strategy_builder, backtest_config, train_size, test_size, step)


def _evaluate_combos(
    combos: list[dict],
    series: CandleSeries,
    strategy_builder,
    backtest_config: BacktestConfig,
    train_size: int,
    test_size: int,
    step,
) -> list[OptimizationResult]:
    k = max(1, len(combos))
    z = _z_for_trials(k)
    be = breakeven_winrate(backtest_config.payout)
    results: list[OptimizationResult] = []

    for params in combos:
        factory = strategy_builder(params)
        report, trades, _ = run_walkforward(series, factory, backtest_config, train_size, test_size, step)
        wins = report.overall.wins
        decisive = wins + report.overall.losses
        if decisive == 0:
            lower = 0.0
        else:
            lower, _ = wilson_interval(wins, decisive, z=z)
        results.append(
            OptimizationResult(
                params=params,
                report=report,
                corrected_has_edge=lower > be,
                z_used=z,
                rank_score=lower,
            )
        )

    results.sort(key=lambda r: r.rank_score, reverse=True)
    return results
