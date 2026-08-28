"""Test crítico: ningún feature de la barra t puede depender del futuro.

Se construyen los features sobre una serie, luego se altera drásticamente
todo lo posterior a un punto de corte `t`, y se vuelve a construir. Si algún
feature de las barras `<= t` cambia, hay una fuga de look-ahead: un bug grave
que invalidaría cualquier backtest o modelo entrenado con este código.
"""

import unittest

from pobot.data.candles import CandleSeries
from pobot.data.synthetic import random_walk_candles
from pobot.features import FEATURE_NAMES, FeatureBuilder
from pobot.types import Candle


class TestNoLookAhead(unittest.TestCase):
    def test_features_unchanged_when_future_is_mutated(self):
        candles = random_walk_candles(120, seed=123)
        series_full = CandleSeries(candles)
        builder = FeatureBuilder()
        rows_full = builder.build(series_full)

        cutoff = 80
        mutated = list(candles[: cutoff + 1])
        # Reemplaza todo lo posterior al cutoff con precios completamente distintos.
        for i in range(cutoff + 1, len(candles)):
            base_ts = candles[i].timestamp
            mutated.append(
                Candle(
                    timestamp=base_ts,
                    open=9999.0 + i,
                    high=10050.0 + i,
                    low=9900.0 + i,
                    close=9975.0 + i,
                    volume=1.0,
                )
            )
        series_mutated = CandleSeries(mutated)
        rows_mutated = builder.build(series_mutated)

        for i in range(cutoff + 1):
            for name in FEATURE_NAMES:
                self.assertAlmostEqual(
                    rows_full[i].values[name],
                    rows_mutated[i].values[name],
                    places=9,
                    msg=f"Look-ahead detectado en feature '{name}' en índice {i}",
                )
            self.assertEqual(rows_full[i].valid, rows_mutated[i].valid)

    def test_truncating_series_does_not_change_past_features(self):
        """Construir features sobre una serie más corta debe dar los mismos
        valores para las barras compartidas (nada usa el largo total de la serie)."""
        candles = random_walk_candles(100, seed=7)
        builder = FeatureBuilder()
        rows_full = builder.build(CandleSeries(candles))
        rows_truncated = builder.build(CandleSeries(candles[:60]))

        for i in range(60):
            for name in FEATURE_NAMES:
                self.assertAlmostEqual(
                    rows_full[i].values[name],
                    rows_truncated[i].values[name],
                    places=9,
                    msg=f"Feature '{name}' en índice {i} depende del largo total de la serie",
                )

    def test_valid_false_until_enough_history(self):
        candles = random_walk_candles(50, seed=5)
        rows = FeatureBuilder().build(CandleSeries(candles))
        self.assertFalse(rows[0].valid)
        self.assertTrue(any(r.valid for r in rows))


if __name__ == "__main__":
    unittest.main()
