import unittest

from pobot.data.candles import CandleSeries
from pobot.labeling import make_labels
from pobot.types import Candle, Direction


def _c(ts, o, h, l, c):
    return Candle(timestamp=ts, open=o, high=h, low=l, close=c, volume=1.0)


class TestLabeling(unittest.TestCase):
    def test_call_when_future_close_above_entry(self):
        candles = [
            _c(0, 10, 10, 10, 10),   # t=0
            _c(60, 10, 10, 10, 10),  # t=1, open usado como entrada de t=0
            _c(120, 10, 12, 9, 12),  # t=2, close usado como salida (horizon=2)
        ]
        series = CandleSeries(candles)
        labels = make_labels(series, horizon=2)
        self.assertEqual(labels.direction[0], Direction.CALL)
        self.assertEqual(labels.entry_price[0], 10.0)
        self.assertEqual(labels.exit_price[0], 12.0)

    def test_put_when_future_close_below_entry(self):
        candles = [
            _c(0, 10, 10, 10, 10),
            _c(60, 10, 10, 10, 10),
            _c(120, 10, 10, 7, 8),
        ]
        series = CandleSeries(candles)
        labels = make_labels(series, horizon=2)
        self.assertEqual(labels.direction[0], Direction.PUT)

    def test_tie_refund_is_none(self):
        candles = [
            _c(0, 10, 10, 10, 10),
            _c(60, 10, 10, 10, 10),
            _c(120, 10, 10, 10, 10),
        ]
        series = CandleSeries(candles)
        labels = make_labels(series, horizon=2, tie_policy="refund")
        self.assertIsNone(labels.direction[0])

    def test_tie_loss_policy_counts_as_put(self):
        candles = [
            _c(0, 10, 10, 10, 10),
            _c(60, 10, 10, 10, 10),
            _c(120, 10, 10, 10, 10),
        ]
        series = CandleSeries(candles)
        labels = make_labels(series, horizon=2, tie_policy="loss")
        self.assertEqual(labels.direction[0], Direction.PUT)

    def test_tail_bars_are_unlabeled(self):
        candles = [_c(i * 60, 10, 11, 9, 10 + i * 0.1) for i in range(10)]
        series = CandleSeries(candles)
        horizon = 3
        labels = make_labels(series, horizon=horizon)
        for t in range(len(series) - horizon, len(series)):
            self.assertIsNone(labels.direction[t])
            self.assertIsNone(labels.entry_price[t])

    def test_invalid_horizon_raises(self):
        series = CandleSeries([_c(0, 1, 1, 1, 1)])
        with self.assertRaises(ValueError):
            make_labels(series, horizon=0)

    def test_invalid_tie_policy_raises(self):
        series = CandleSeries([_c(0, 1, 1, 1, 1)])
        with self.assertRaises(ValueError):
            make_labels(series, horizon=1, tie_policy="bogus")


if __name__ == "__main__":
    unittest.main()
