import unittest

from pobot.indicators.core import ema, rma, rolling_max, rolling_min, rolling_std, sma, true_range
from pobot.indicators.momentum import macd, roc, rsi, stochastic
from pobot.indicators.patterns import (
    body_to_range_ratio,
    is_doji,
    is_engulfing,
    is_pin_bar,
)
from pobot.indicators.trend import adx, donchian_channel, ema_slope
from pobot.indicators.volatility import atr, bollinger_bands
from pobot.indicators.volume import obv, session_vwap, volume_zscore
from pobot.types import Candle, Direction


def _candle(ts, o, h, l, c, v=100.0):
    return Candle(timestamp=ts, open=o, high=h, low=l, close=c, volume=v)


class TestCore(unittest.TestCase):
    def test_sma_basic(self):
        values = [1, 2, 3, 4, 5]
        result = sma(values, 3)
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])
        self.assertAlmostEqual(result[2], 2.0)
        self.assertAlmostEqual(result[3], 3.0)
        self.assertAlmostEqual(result[4], 4.0)

    def test_ema_seeds_with_sma_then_recurses(self):
        values = [1, 2, 3, 4, 5, 6, 7]
        period = 3
        result = ema(values, period)
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])
        self.assertAlmostEqual(result[2], 2.0)  # seed = SMA(1,2,3)
        alpha = 2 / (period + 1)
        expected_3 = alpha * 4 + (1 - alpha) * 2.0
        self.assertAlmostEqual(result[3], expected_3, places=9)

    def test_rma_matches_wilder_seed(self):
        values = [1, 2, 3, 4, 5]
        result = rma(values, 3)
        self.assertAlmostEqual(result[2], 2.0)
        expected_3 = (2.0 * 2 + 4) / 3
        self.assertAlmostEqual(result[3], expected_3, places=9)

    def test_rolling_std_known_values(self):
        # [2, 4, 4, 4, 5, 5, 7, 9] tiene std poblacional = 2.0 (ejemplo clásico)
        values = [2, 4, 4, 4, 5, 5, 7, 9]
        result = rolling_std(values, 8)
        self.assertAlmostEqual(result[7], 2.0, places=6)

    def test_true_range_first_bar_is_high_minus_low(self):
        candles = [_candle(0, 10, 12, 9, 11), _candle(60, 11, 15, 10, 14)]
        tr = true_range(candles)
        self.assertAlmostEqual(tr[0], 3.0)
        # max(15-10, |15-11|, |10-11|) = max(5, 4, 1) = 5
        self.assertAlmostEqual(tr[1], 5.0)

    def test_rolling_min_max_monotonic_deque(self):
        values = [5, 3, 8, 1, 9, 2, 7]
        mins = rolling_min(values, 3)
        maxs = rolling_max(values, 3)
        # ventana [5,3,8] -> min 3, max 8 en índice 2
        self.assertEqual(mins[2], 3)
        self.assertEqual(maxs[2], 8)
        # ventana [8,1,9] -> min 1, max 9 en índice 4
        self.assertEqual(mins[4], 1)
        self.assertEqual(maxs[4], 9)


class TestMomentum(unittest.TestCase):
    def test_rsi_all_gains_is_100(self):
        closes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        result = rsi(closes, period=14)
        self.assertAlmostEqual(result[-1], 100.0)

    def test_rsi_all_losses_is_0(self):
        closes = list(range(20, 0, -1))
        result = rsi(closes, period=14)
        self.assertAlmostEqual(result[-1], 0.0)

    def test_rsi_bounded_0_100_on_mixed_series(self):
        closes = [10, 11, 9, 12, 8, 13, 7, 14, 6, 15, 20, 5, 25, 4, 30, 3, 35, 2]
        result = rsi(closes, period=5)
        for v in result:
            if v is not None:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 100.0)

    def test_macd_histogram_is_macd_minus_signal(self):
        closes = [float(x) for x in list(range(1, 60))]
        result = macd(closes, fast=12, slow=26, signal_period=9)
        for m, s, h in zip(result.macd, result.signal, result.histogram):
            if m is not None and s is not None:
                self.assertAlmostEqual(h, m - s, places=9)

    def test_stochastic_bounded_0_100(self):
        highs = [float(x) for x in [10, 12, 11, 14, 13, 15, 16, 12, 11, 18, 20, 9, 8, 22, 21]]
        lows = [h - 2 for h in highs]
        closes = [h - 1 for h in highs]
        k, d = stochastic(highs, lows, closes, period=5, smooth_k=3)
        for v in k + d:
            if v is not None:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 100.0)

    def test_roc_known_value(self):
        closes = [100, 100, 100, 100, 110]
        result = roc(closes, period=4)
        self.assertAlmostEqual(result[4], 10.0)


class TestVolatility(unittest.TestCase):
    def test_atr_first_valid_equals_average_true_range(self):
        candles = [
            _candle(i * 60, 10 + i, 12 + i, 9 + i, 11 + i) for i in range(20)
        ]
        result = atr(candles, period=14)
        # true_range() ya da un valor válido en la barra 0 (high - low), así
        # que el ATR de periodo 14 se vuelve válido en el índice 13 (0-based).
        self.assertIsNotNone(result[13])
        self.assertIsNone(result[12])

    def test_bollinger_upper_above_lower(self):
        closes = [float(x) for x in [10, 11, 9, 12, 8, 13, 7, 14, 6, 15, 20, 5, 25, 4, 30]]
        bands = bollinger_bands(closes, period=5, num_std=2.0)
        for u, l in zip(bands.upper, bands.lower):
            if u is not None:
                self.assertGreaterEqual(u, l)

    def test_bollinger_middle_equals_sma(self):
        from pobot.indicators.core import sma

        closes = [float(x) for x in range(1, 30)]
        bands = bollinger_bands(closes, period=10)
        expected = sma(closes, 10)
        for a, b in zip(bands.middle, expected):
            if a is not None:
                self.assertAlmostEqual(a, b, places=9)


class TestTrend(unittest.TestCase):
    def test_adx_bounded_0_100(self):
        candles = [
            _candle(i * 60, 10 + i * 0.5, 11 + i * 0.5, 9 + i * 0.5, 10.5 + i * 0.5)
            for i in range(40)
        ]
        result = adx(candles, period=14)
        for v in result.adx:
            if v is not None:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 100.0)

    def test_donchian_channel_contains_price(self):
        highs = [float(x) for x in [10, 12, 11, 14, 13, 15, 16, 12, 11, 18]]
        lows = [h - 3 for h in highs]
        result = donchian_channel(highs, lows, period=5)
        for i in range(4, len(highs)):
            self.assertGreaterEqual(result.upper[i], max(highs[i - 4 : i + 1]) - 1e-9)
            self.assertLessEqual(result.lower[i], min(lows[i - 4 : i + 1]) + 1e-9)

    def test_ema_slope_positive_on_uptrend(self):
        closes = [float(x) for x in range(1, 50)]
        slope = ema_slope(closes, period=10, lookback=3)
        valid = [s for s in slope if s is not None]
        self.assertTrue(all(s > 0 for s in valid))

    def test_ema_slope_negative_on_downtrend(self):
        closes = [float(x) for x in range(50, 1, -1)]
        slope = ema_slope(closes, period=10, lookback=3)
        valid = [s for s in slope if s is not None]
        self.assertTrue(all(s < 0 for s in valid))


class TestVolume(unittest.TestCase):
    def test_obv_increases_on_up_candle(self):
        candles = [_candle(0, 10, 11, 9, 10, v=100), _candle(60, 10, 12, 9, 11, v=50)]
        result = obv(candles)
        self.assertEqual(result[0], 0.0)
        self.assertEqual(result[1], 50.0)

    def test_obv_decreases_on_down_candle(self):
        candles = [_candle(0, 10, 11, 9, 10, v=100), _candle(60, 10, 11, 8, 9, v=50)]
        result = obv(candles)
        self.assertEqual(result[1], -50.0)

    def test_session_vwap_resets_on_new_session(self):
        day = 86400
        candles = [
            _candle(0, 10, 11, 9, 10, v=10),
            _candle(60, 10, 11, 9, 10, v=10),
            _candle(day, 100, 101, 99, 100, v=10),
        ]
        result = session_vwap(candles, session_seconds=day)
        self.assertAlmostEqual(result[2], 100.0, places=6)

    def test_volume_zscore_zero_for_constant_volume(self):
        candles = [_candle(i * 60, 10, 11, 9, 10, v=100.0) for i in range(10)]
        result = volume_zscore(candles, period=5)
        for v in result:
            if v is not None:
                self.assertAlmostEqual(v, 0.0, places=9)


class TestPatterns(unittest.TestCase):
    def test_doji_detects_tiny_body(self):
        candles = [_candle(0, 10.0, 11.0, 9.0, 10.05)]
        self.assertTrue(is_doji(candles, threshold=0.1)[0])

    def test_not_doji_on_big_body(self):
        candles = [_candle(0, 10.0, 11.0, 9.0, 10.9)]
        self.assertFalse(is_doji(candles, threshold=0.1)[0])

    def test_pin_bar_bullish_on_long_lower_wick(self):
        # cuerpo pequeño arriba, mecha inferior larga
        c = _candle(0, 10.0, 10.2, 8.0, 10.1)
        result = is_pin_bar([c], wick_ratio=2.0)
        self.assertEqual(result[0], Direction.CALL)

    def test_pin_bar_bearish_on_long_upper_wick(self):
        c = _candle(0, 10.0, 12.0, 9.9, 10.1)
        result = is_pin_bar([c], wick_ratio=2.0)
        self.assertEqual(result[0], Direction.PUT)

    def test_bullish_engulfing_detected(self):
        prev = _candle(0, 10.0, 10.1, 8.9, 9.0)  # bajista
        curr = _candle(60, 8.8, 11.0, 8.7, 10.5)  # alcista, cubre el cuerpo previo
        result = is_engulfing([prev, curr])
        self.assertEqual(result[1], Direction.CALL)

    def test_bearish_engulfing_detected(self):
        prev = _candle(0, 9.0, 10.1, 8.9, 10.0)  # alcista
        curr = _candle(60, 10.2, 10.3, 8.5, 8.6)  # bajista, cubre el cuerpo previo
        result = is_engulfing([prev, curr])
        self.assertEqual(result[1], Direction.PUT)

    def test_body_to_range_ratio_zero_range_is_zero(self):
        c = _candle(0, 10.0, 10.0, 10.0, 10.0)
        self.assertEqual(body_to_range_ratio([c])[0], 0.0)


if __name__ == "__main__":
    unittest.main()
