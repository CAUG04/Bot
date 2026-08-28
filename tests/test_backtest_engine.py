import unittest

from pobot.backtest.engine import BacktestConfig, run_backtest
from pobot.data.candles import CandleSeries
from pobot.strategies.base import Strategy
from pobot.types import Candle, Direction, Signal, TradeResult


def _c(ts, o, h, l, c):
    return Candle(timestamp=ts, open=o, high=h, low=l, close=c, volume=1.0)


class AlwaysCallStrategy(Strategy):
    name = "always_call_test"

    def __init__(self, expiry_bars: int = 1, confidence: float = 0.6):
        self.expiry_bars = expiry_bars
        self.confidence = confidence

    def evaluate(self, series, i):
        return Signal(
            index=i,
            timestamp=series[i].timestamp,
            direction=Direction.CALL,
            confidence=self.confidence,
            expiry_bars=self.expiry_bars,
            strategy=self.name,
        )


class EveryOtherHourStrategy(Strategy):
    """Solo emite señal si la vela de entrada cae en hora par (para probar allowed_hours)."""

    name = "every_other_hour_test"

    def evaluate(self, series, i):
        return Signal(
            index=i, timestamp=series[i].timestamp, direction=Direction.CALL,
            confidence=0.6, expiry_bars=1, strategy=self.name,
        )


class TestBacktestEngine(unittest.TestCase):
    def test_call_wins_when_price_rises(self):
        candles = [
            _c(0, 10, 10, 10, 10),
            _c(60, 10, 10, 10, 10),   # entrada al open de esta vela: 10
            _c(120, 10, 12, 9, 12),   # salida al close: 12 -> sube -> CALL gana
        ]
        series = CandleSeries(candles)
        strategy = AlwaysCallStrategy(expiry_bars=2)
        report = run_backtest(series, strategy, BacktestConfig(payout=0.85))
        self.assertEqual(report.n_trades, 1)
        self.assertEqual(report.trades[0].result, TradeResult.WIN)

    def test_call_loses_when_price_falls(self):
        candles = [
            _c(0, 10, 10, 10, 10),
            _c(60, 10, 10, 10, 10),
            _c(120, 10, 10, 7, 8),
        ]
        series = CandleSeries(candles)
        strategy = AlwaysCallStrategy(expiry_bars=2)
        report = run_backtest(series, strategy, BacktestConfig(payout=0.85))
        self.assertEqual(report.trades[0].result, TradeResult.LOSS)

    def test_no_trade_when_insufficient_future_bars(self):
        candles = [_c(i * 60, 10, 10, 10, 10) for i in range(3)]
        series = CandleSeries(candles)
        strategy = AlwaysCallStrategy(expiry_bars=5)  # excede el largo de la serie
        report = run_backtest(series, strategy, BacktestConfig(payout=0.85))
        self.assertEqual(report.n_trades, 0)
        self.assertGreater(report.signals_filtered, 0)

    def test_cooldown_blocks_immediate_next_signal(self):
        candles = [_c(i * 60, 10, 10 + i * 0.01, 10 - i * 0.01, 10 + i * 0.005) for i in range(20)]
        series = CandleSeries(candles)
        strategy = AlwaysCallStrategy(expiry_bars=1)
        report_no_cooldown = run_backtest(series, strategy, BacktestConfig(payout=0.85, cooldown_bars=1))
        report_cooldown = run_backtest(series, strategy, BacktestConfig(payout=0.85, cooldown_bars=5))
        self.assertGreater(report_no_cooldown.n_trades, report_cooldown.n_trades)

    def test_max_trades_per_day_limits_trades(self):
        # 30 velas de 1 minuto, todas en el mismo día UTC
        candles = [_c(i * 60, 10, 10.1, 9.9, 10.05) for i in range(30)]
        series = CandleSeries(candles)
        strategy = AlwaysCallStrategy(expiry_bars=1)
        config = BacktestConfig(payout=0.85, cooldown_bars=1, max_trades_per_day=3)
        report = run_backtest(series, strategy, config)
        self.assertLessEqual(report.n_trades, 3)

    def test_allowed_hours_filters_out_other_hours(self):
        # timestamp 0 = hora 0 UTC; generamos velas en varias horas distintas
        candles = [_c(h * 3600, 10, 10.1, 9.9, 10.05) for h in range(24)]
        candles.append(_c(24 * 3600, 10, 10.1, 9.9, 10.05))  # vela de salida final
        series = CandleSeries(candles)
        strategy = AlwaysCallStrategy(expiry_bars=1)
        config = BacktestConfig(payout=0.85, allowed_hours=[10])
        report = run_backtest(series, strategy, config)
        for t in report.trades:
            entry_hour = (t.entry_timestamp // 3600) % 24
            self.assertEqual(entry_hour, 10)

    def test_tie_policy_refund_gives_zero_pnl(self):
        candles = [
            _c(0, 10, 10, 10, 10),
            _c(60, 10, 10, 10, 10),
            _c(120, 10, 10, 10, 10),  # close == entry -> empate
        ]
        series = CandleSeries(candles)
        strategy = AlwaysCallStrategy(expiry_bars=2)
        report = run_backtest(series, strategy, BacktestConfig(payout=0.85, tie_policy="refund"))
        self.assertEqual(report.trades[0].result, TradeResult.TIE)
        self.assertEqual(report.trades[0].pnl, 0.0)

    def test_tie_policy_loss_forces_loss_regardless_of_direction(self):
        candles = [
            _c(0, 10, 10, 10, 10),
            _c(60, 10, 10, 10, 10),
            _c(120, 10, 10, 10, 10),
        ]
        series = CandleSeries(candles)
        strategy = AlwaysCallStrategy(expiry_bars=2)
        report = run_backtest(series, strategy, BacktestConfig(payout=0.85, tie_policy="loss"))
        self.assertEqual(report.trades[0].result, TradeResult.LOSS)
        self.assertEqual(report.trades[0].pnl, -1.0 * report.trades[0].stake)


if __name__ == "__main__":
    unittest.main()
