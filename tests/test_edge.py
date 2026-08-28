import unittest

from pobot.edge import (
    breakeven_winrate,
    expected_value,
    wilson_interval,
    has_demonstrated_edge,
    kelly_fraction,
)


class TestEdgeMath(unittest.TestCase):
    def test_breakeven_winrate_92(self):
        self.assertAlmostEqual(breakeven_winrate(0.92), 1 / 1.92, places=6)

    def test_breakeven_winrate_80(self):
        self.assertAlmostEqual(breakeven_winrate(0.80), 1 / 1.80, places=6)

    def test_expected_value_at_breakeven_is_zero(self):
        payout = 0.87
        w = breakeven_winrate(payout)
        self.assertAlmostEqual(expected_value(w, payout), 0.0, places=9)

    def test_expected_value_above_breakeven_positive(self):
        payout = 0.85
        w = breakeven_winrate(payout) + 0.05
        self.assertGreater(expected_value(w, payout), 0.0)

    def test_expected_value_below_breakeven_negative(self):
        payout = 0.85
        w = breakeven_winrate(payout) - 0.05
        self.assertLess(expected_value(w, payout), 0.0)

    def test_wilson_interval_contains_point_estimate(self):
        lower, upper = wilson_interval(60, 100)
        self.assertLessEqual(lower, 0.60)
        self.assertGreaterEqual(upper, 0.60)

    def test_wilson_interval_narrows_with_more_data(self):
        lo_small, hi_small = wilson_interval(60, 100)
        lo_big, hi_big = wilson_interval(6000, 10000)
        self.assertLess(hi_big - lo_big, hi_small - lo_small)

    def test_wilson_interval_bounds(self):
        lower, upper = wilson_interval(0, 10)
        self.assertGreaterEqual(lower, 0.0)
        lower, upper = wilson_interval(10, 10)
        self.assertLessEqual(upper, 1.0)

    def test_no_edge_with_small_sample_even_if_winrate_high(self):
        # 7 de 10 aciertos "parece" ganador, pero con n=10 no es concluyente.
        self.assertFalse(has_demonstrated_edge(7, 10, payout=0.85))

    def test_edge_demonstrated_with_large_sample_above_breakeven(self):
        # 58% de winrate sobre 5000 operaciones con payout 0.85 (breakeven ~54.05%)
        n = 5000
        wins = int(0.58 * n)
        self.assertTrue(has_demonstrated_edge(wins, n, payout=0.85))

    def test_no_edge_at_random_50_50(self):
        n = 5000
        wins = n // 2
        self.assertFalse(has_demonstrated_edge(wins, n, payout=0.85))

    def test_kelly_fraction_zero_when_no_edge(self):
        self.assertEqual(kelly_fraction(0.5, 0.85), 0.0)

    def test_kelly_fraction_positive_with_edge(self):
        f = kelly_fraction(0.60, 0.85)
        self.assertGreater(f, 0.0)

    def test_kelly_fraction_capped(self):
        f = kelly_fraction(0.95, 5.0, cap=0.25)
        self.assertLessEqual(f, 0.25)


if __name__ == "__main__":
    unittest.main()
