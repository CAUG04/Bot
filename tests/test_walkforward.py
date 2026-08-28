import unittest

from pobot.backtest.engine import BacktestConfig
from pobot.backtest.walkforward import make_windows, run_walkforward
from pobot.data.candles import CandleSeries
from pobot.data.synthetic import random_walk_candles
from pobot.strategies.base import Strategy
from pobot.types import Direction, Signal


class _AlwaysCall(Strategy):
    name = "always_call_wf_test"

    def evaluate(self, series, i):
        return Signal(
            index=i, timestamp=series[i].timestamp, direction=Direction.CALL,
            confidence=0.6, expiry_bars=1, strategy=self.name,
        )


class TestMakeWindows(unittest.TestCase):
    def test_windows_respect_train_test_sizes(self):
        windows = make_windows(n=200, train_size=50, test_size=20)
        for w in windows:
            self.assertEqual(w.train_end - w.train_start, 50)
            self.assertEqual(w.test_end - w.test_start, 20)
            self.assertEqual(w.train_end, w.test_start)

    def test_windows_do_not_exceed_series_length(self):
        n = 200
        windows = make_windows(n=n, train_size=50, test_size=20)
        for w in windows:
            self.assertLessEqual(w.test_end, n)

    def test_no_overlap_between_consecutive_test_windows_by_default(self):
        windows = make_windows(n=300, train_size=50, test_size=20)
        for a, b in zip(windows, windows[1:]):
            self.assertLessEqual(a.test_end, b.test_start)

    def test_invalid_sizes_raise(self):
        with self.assertRaises(ValueError):
            make_windows(100, train_size=0, test_size=10)


class TestRunWalkforward(unittest.TestCase):
    def test_trades_fall_within_test_windows_only(self):
        candles = random_walk_candles(300, seed=11)
        series = CandleSeries(candles)

        def factory(train_series):
            return _AlwaysCall()

        config = BacktestConfig(payout=0.85, cooldown_bars=1)
        report, trades, windows = run_walkforward(series, factory, config, train_size=50, test_size=30)

        self.assertGreater(len(trades), 0)
        test_ranges = [(w.test_start, w.test_end) for w in windows]
        for t in trades:
            signal_index = t.entry_index - 1
            self.assertTrue(
                any(start <= signal_index < end for start, end in test_ranges),
                f"trade con señal en índice {signal_index} cae fuera de toda ventana de test",
            )

    def test_report_is_consistent_with_trade_count(self):
        candles = random_walk_candles(300, seed=22)
        series = CandleSeries(candles)

        def factory(train_series):
            return _AlwaysCall()

        config = BacktestConfig(payout=0.85)
        report, trades, _ = run_walkforward(series, factory, config, train_size=40, test_size=20)
        self.assertEqual(report.overall.n_trades, len(trades))


if __name__ == "__main__":
    unittest.main()
