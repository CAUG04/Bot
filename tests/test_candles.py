import tempfile
import unittest
from pathlib import Path

from pobot.data.candles import CandleSeries
from pobot.data.synthetic import random_walk_candles
from pobot.types import Candle


class TestCandleSeries(unittest.TestCase):
    def test_rejects_non_monotonic_timestamps(self):
        candles = [
            Candle(timestamp=100, open=1, high=1, low=1, close=1),
            Candle(timestamp=100, open=1, high=1, low=1, close=1),
        ]
        with self.assertRaises(ValueError):
            CandleSeries(candles)

    def test_csv_roundtrip(self):
        candles = random_walk_candles(50)
        series = CandleSeries(candles)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "candles.csv"
            series.to_csv(path)
            loaded = CandleSeries.from_csv(path)
        self.assertEqual(len(loaded), len(series))
        for a, b in zip(series, loaded):
            self.assertEqual(a.timestamp, b.timestamp)
            self.assertAlmostEqual(a.close, b.close, places=6)

    def test_check_gaps_detects_missing_candle(self):
        candles = random_walk_candles(10, step_seconds=60)
        # Elimina la vela de en medio para forzar un hueco de 120s
        with_gap = candles[:5] + candles[6:]
        series = CandleSeries(with_gap)
        report = series.check_gaps(expected_step=60)
        self.assertTrue(report.has_gaps)
        self.assertEqual(len(report.gaps), 1)

    def test_no_gaps_on_clean_series(self):
        series = CandleSeries(random_walk_candles(20, step_seconds=60))
        report = series.check_gaps(expected_step=60)
        self.assertFalse(report.has_gaps)

    def test_resample_groups_correctly(self):
        candles = random_walk_candles(10, step_seconds=60)
        series = CandleSeries(candles)
        resampled = series.resample(5)
        self.assertEqual(len(resampled), 2)
        first_group = candles[:5]
        r0 = resampled[0]
        self.assertEqual(r0.open, first_group[0].open)
        self.assertEqual(r0.close, first_group[-1].close)
        self.assertEqual(r0.high, max(c.high for c in first_group))
        self.assertEqual(r0.low, min(c.low for c in first_group))
        self.assertEqual(r0.timestamp, first_group[-1].timestamp)

    def test_resample_discards_incomplete_tail(self):
        candles = random_walk_candles(12, step_seconds=60)
        series = CandleSeries(candles)
        resampled = series.resample(5)
        self.assertEqual(len(resampled), 2)  # 12 // 5 = 2, sobran 2 velas descartadas

    def test_from_csv_missing_column_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.csv"
            path.write_text("timestamp,open,high,low,close\n1,2,3,4,5\n")
            with self.assertRaises(ValueError):
                CandleSeries.from_csv(path)


if __name__ == "__main__":
    unittest.main()
