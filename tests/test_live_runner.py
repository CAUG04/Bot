import unittest

from pobot.data.candles import CandleSeries
from pobot.live.runner import LiveRunner, LiveRunnerConfig
from pobot.strategies.base import Strategy
from pobot.types import Candle, Direction, Signal


def _c(ts, o, h, l, c):
    return Candle(timestamp=ts, open=o, high=h, low=l, close=c, volume=1.0)


class _FakeNotifier:
    def __init__(self):
        self.messages: list[str] = []

    def send(self, message: str) -> None:
        self.messages.append(message)


class _AlwaysCallStrategy(Strategy):
    name = "always_call_runner_test"

    def evaluate(self, series, i):
        return Signal(
            index=i, timestamp=series[i].timestamp, direction=Direction.CALL,
            confidence=0.6, expiry_bars=1, strategy=self.name,
        )


class _NeverSignalStrategy(Strategy):
    name = "never_signal_runner_test"

    def evaluate(self, series, i):
        return None


class TestLiveRunnerStep(unittest.TestCase):
    def _series(self, n=10):
        return CandleSeries([_c(i * 60, 10, 11, 9, 10 + i * 0.1) for i in range(n)])

    def test_emits_signal_and_notifies(self):
        series = self._series()
        notifier = _FakeNotifier()
        runner = LiveRunner(
            _AlwaysCallStrategy(), notifier, lambda: series,
            LiveRunnerConfig(symbol="BTCUSDT", interval="1m", payout=0.85),
        )
        signal = runner.step()
        self.assertIsNotNone(signal)
        self.assertEqual(len(notifier.messages), 1)
        self.assertIn("BTCUSDT", notifier.messages[0])

    def test_no_notification_when_strategy_has_no_signal(self):
        series = self._series()
        notifier = _FakeNotifier()
        runner = LiveRunner(
            _NeverSignalStrategy(), notifier, lambda: series,
            LiveRunnerConfig(symbol="BTCUSDT", interval="1m", payout=0.85),
        )
        signal = runner.step()
        self.assertIsNone(signal)
        self.assertEqual(len(notifier.messages), 0)

    def test_does_not_duplicate_signal_on_same_closed_candle(self):
        series = self._series()
        notifier = _FakeNotifier()
        runner = LiveRunner(
            _AlwaysCallStrategy(), notifier, lambda: series,
            LiveRunnerConfig(symbol="BTCUSDT", interval="1m", payout=0.85),
        )
        runner.step()
        runner.step()  # misma vela, no debe volver a notificar
        self.assertEqual(len(notifier.messages), 1)

    def test_new_candle_triggers_new_signal(self):
        series = self._series(10)
        notifier = _FakeNotifier()
        state = {"series": series}
        runner = LiveRunner(
            _AlwaysCallStrategy(), notifier, lambda: state["series"],
            LiveRunnerConfig(symbol="BTCUSDT", interval="1m", payout=0.85),
        )
        runner.step()
        state["series"] = self._series(11)
        runner.step()
        self.assertEqual(len(notifier.messages), 2)

    def test_allowed_hours_filters_signal(self):
        series = self._series()
        notifier = _FakeNotifier()
        last_hour = (series[len(series) - 1].timestamp // 3600) % 24
        other_hour = (last_hour + 5) % 24
        runner = LiveRunner(
            _AlwaysCallStrategy(), notifier, lambda: series,
            LiveRunnerConfig(symbol="BTCUSDT", interval="1m", payout=0.85, allowed_hours=[other_hour]),
        )
        signal = runner.step()
        self.assertIsNone(signal)
        self.assertEqual(len(notifier.messages), 0)

    def test_insufficient_history_returns_none(self):
        series = self._series(2)

        class _WarmupStrategy(Strategy):
            name = "warmup_test"

            def warmup(self):
                return 50

            def evaluate(self, series, i):
                return Signal(index=i, timestamp=series[i].timestamp, direction=Direction.CALL,
                               confidence=0.6, expiry_bars=1, strategy=self.name)

        notifier = _FakeNotifier()
        runner = LiveRunner(
            _WarmupStrategy(), notifier, lambda: series,
            LiveRunnerConfig(symbol="BTCUSDT", interval="1m", payout=0.85),
        )
        self.assertIsNone(runner.step())

    def test_run_forever_respects_max_iterations(self):
        series = self._series()
        notifier = _FakeNotifier()
        runner = LiveRunner(
            _NeverSignalStrategy(), notifier, lambda: series,
            LiveRunnerConfig(symbol="BTCUSDT", interval="1m", payout=0.85, poll_seconds=0.0),
        )
        runner.run_forever(max_iterations=3)  # no debe colgarse ni lanzar

    def test_step_survives_strategy_exception_in_run_forever(self):
        class _BoomStrategy(Strategy):
            name = "boom_test"

            def evaluate(self, series, i):
                raise RuntimeError("boom")

        series = self._series()
        notifier = _FakeNotifier()
        runner = LiveRunner(
            _BoomStrategy(), notifier, lambda: series,
            LiveRunnerConfig(symbol="BTCUSDT", interval="1m", payout=0.85, poll_seconds=0.0),
        )
        runner.run_forever(max_iterations=2)  # no debe propagar la excepción


if __name__ == "__main__":
    unittest.main()
