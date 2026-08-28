import unittest

from pobot.backtest.metrics import compute_report
from pobot.types import Direction, Trade


def _trade(entry_ts, direction, entry_price, exit_price, expiry_bars=1, payout=0.85, stake=1.0):
    return Trade(
        entry_index=0,
        entry_timestamp=entry_ts,
        exit_index=0,
        exit_timestamp=entry_ts + 60 * expiry_bars,
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        expiry_bars=expiry_bars,
        stake=stake,
        payout=payout,
        strategy="test",
    )


class TestMetrics(unittest.TestCase):
    def test_all_wins_winrate_100(self):
        trades = [_trade(0, Direction.CALL, 10, 11) for _ in range(20)]
        report = compute_report(trades, payout=0.85)
        self.assertEqual(report.overall.wins, 20)
        self.assertAlmostEqual(report.overall.winrate, 1.0)
        self.assertTrue(report.overall.has_edge)

    def test_all_losses_no_edge(self):
        trades = [_trade(0, Direction.CALL, 10, 9) for _ in range(20)]
        report = compute_report(trades, payout=0.85)
        self.assertEqual(report.overall.losses, 20)
        self.assertFalse(report.overall.has_edge)

    def test_ties_excluded_from_winrate_denominator(self):
        trades = [_trade(0, Direction.CALL, 10, 11)] * 5 + [_trade(0, Direction.CALL, 10, 10)] * 5
        report = compute_report(trades, payout=0.85)
        self.assertEqual(report.overall.wins, 5)
        self.assertEqual(report.overall.ties, 5)
        self.assertAlmostEqual(report.overall.winrate, 1.0)  # 5 wins / (5 wins + 0 losses)

    def test_profit_factor_infinite_with_no_losses(self):
        trades = [_trade(0, Direction.CALL, 10, 11) for _ in range(5)]
        report = compute_report(trades, payout=0.85)
        self.assertEqual(report.profit_factor, float("inf"))

    def test_by_hour_segments_correctly(self):
        trades = [
            _trade(0 * 3600, Direction.CALL, 10, 11),      # hora 0
            _trade(5 * 3600, Direction.CALL, 10, 9),       # hora 5
            _trade(24 * 3600, Direction.CALL, 10, 11),     # hora 0 (día siguiente)
        ]
        report = compute_report(trades, payout=0.85)
        self.assertEqual(report.by_hour[0].n_trades, 2)
        self.assertEqual(report.by_hour[5].n_trades, 1)

    def test_by_expiry_segments_correctly(self):
        trades = [
            _trade(0, Direction.CALL, 10, 11, expiry_bars=1),
            _trade(0, Direction.CALL, 10, 11, expiry_bars=3),
            _trade(0, Direction.CALL, 10, 11, expiry_bars=3),
        ]
        report = compute_report(trades, payout=0.85)
        self.assertEqual(report.by_expiry[1].n_trades, 1)
        self.assertEqual(report.by_expiry[3].n_trades, 2)

    def test_drawdown_tracks_losing_streak(self):
        trades = (
            [_trade(0, Direction.CALL, 10, 11)] * 2
            + [_trade(0, Direction.CALL, 10, 9)] * 4
            + [_trade(0, Direction.CALL, 10, 11)] * 1
        )
        report = compute_report(trades, payout=0.85)
        self.assertEqual(report.drawdown.longest_losing_streak, 4)
        self.assertEqual(report.drawdown.longest_winning_streak, 2)
        self.assertGreater(report.drawdown.max_drawdown, 0)

    def test_empty_trades_does_not_crash(self):
        report = compute_report([], payout=0.85)
        self.assertEqual(report.overall.n_trades, 0)
        self.assertFalse(report.overall.has_edge)
        self.assertEqual(report.profit_factor, 0.0)

    def test_summary_lines_produces_readable_text(self):
        trades = [_trade(0, Direction.CALL, 10, 11) for _ in range(10)]
        report = compute_report(trades, payout=0.85)
        lines = report.summary_lines()
        self.assertGreater(len(lines), 3)
        self.assertTrue(any("Winrate" in l for l in lines))


if __name__ == "__main__":
    unittest.main()
