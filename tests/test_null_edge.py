"""El test más importante del repositorio.

Sobre un random walk sin estructura explotable (drift=0, ruido gaussiano puro),
NINGUNA estrategia sin información real del futuro debería poder demostrar
ventaja estadística. Si este test falla, hay un bug de look-ahead o de
contabilidad en el motor de backtest — no una "buena estrategia" — porque por
construcción el generador sintético no tiene ninguna señal predecible.

Se prueban varias estrategias "ingenuas" (siempre CALL, siempre PUT,
alternando, basada en paridad del índice) para cubrir sesgos triviales que un
motor con bugs podría filtrar accidentalmente.
"""

import unittest

from pobot.backtest.engine import BacktestConfig, run_backtest
from pobot.data.candles import CandleSeries
from pobot.data.synthetic import random_walk_candles
from pobot.edge import breakeven_winrate
from pobot.strategies.base import Strategy
from pobot.types import Direction, Signal

N_CANDLES = 4000
PAYOUT = 0.85


class _FixedDirectionStrategy(Strategy):
    def __init__(self, direction: Direction, expiry_bars: int = 1):
        self.direction = direction
        self.expiry_bars = expiry_bars
        self.name = f"fixed_{direction.value}_{expiry_bars}"

    def evaluate(self, series, i):
        return Signal(
            index=i, timestamp=series[i].timestamp, direction=self.direction,
            confidence=0.6, expiry_bars=self.expiry_bars, strategy=self.name,
        )


class _AlternatingStrategy(Strategy):
    name = "alternating_test"

    def evaluate(self, series, i):
        direction = Direction.CALL if i % 2 == 0 else Direction.PUT
        return Signal(
            index=i, timestamp=series[i].timestamp, direction=direction,
            confidence=0.6, expiry_bars=1, strategy=self.name,
        )


class _ParityBasedStrategy(Strategy):
    """Decide dirección según si el índice de barra es múltiplo de 3.
    Sesgo arbitrario, sin relación causal con el precio futuro."""

    name = "parity_test"

    def evaluate(self, series, i):
        direction = Direction.CALL if i % 3 == 0 else Direction.PUT
        return Signal(
            index=i, timestamp=series[i].timestamp, direction=direction,
            confidence=0.6, expiry_bars=2, strategy=self.name,
        )


class TestNullEdgeOnPureNoise(unittest.TestCase):
    def setUp(self):
        self.series = CandleSeries(random_walk_candles(N_CANDLES, drift=0.0, seed=999))
        self.config = BacktestConfig(payout=PAYOUT, cooldown_bars=1)

    def _assert_no_edge(self, strategy):
        report = run_backtest(self.series, strategy, self.config)
        self.assertGreater(report.n_trades, 100, "necesitamos suficientes trades para que el test sea informativo")

        from pobot.backtest.metrics import compute_report

        full = compute_report(report.trades, PAYOUT)
        self.assertFalse(
            full.overall.has_edge,
            f"¡Se detectó ventaja falsa en ruido puro! winrate={full.overall.winrate:.4f}, "
            f"IC=[{full.overall.wilson_lower:.4f}, {full.overall.wilson_upper:.4f}], "
            f"breakeven={breakeven_winrate(PAYOUT):.4f}. Esto indica un bug de look-ahead "
            f"o de contabilidad en el motor de backtest.",
        )
        # Además, el winrate puntual debe estar razonablemente cerca de 50%
        # (una desviación grande también delataría un bug de contabilidad).
        self.assertAlmostEqual(full.overall.winrate, 0.5, delta=0.06)

    def test_always_call_has_no_edge(self):
        self._assert_no_edge(_FixedDirectionStrategy(Direction.CALL, expiry_bars=1))

    def test_always_put_has_no_edge(self):
        self._assert_no_edge(_FixedDirectionStrategy(Direction.PUT, expiry_bars=3))

    def test_alternating_direction_has_no_edge(self):
        self._assert_no_edge(_AlternatingStrategy())

    def test_arbitrary_parity_rule_has_no_edge(self):
        self._assert_no_edge(_ParityBasedStrategy())


if __name__ == "__main__":
    unittest.main()
