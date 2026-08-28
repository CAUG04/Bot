"""Construcción de vectores de features por barra, sin look-ahead.

Cada indicador subyacente ya garantiza que el valor en el índice `i` depende
solo de `candles[0..i]` (ver `pobot/indicators/`). `FeatureBuilder` combina
esos indicadores en un vector numérico por barra, normalizado para que sea
comparable entre activos y regímenes de volatilidad.

Filas con algún feature en `None` (historial insuficiente) se marcan como
`valid=False` y deben excluirse del entrenamiento/backtest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pobot.data.candles import CandleSeries
from pobot.indicators.core import ema
from pobot.indicators.momentum import macd, rsi, stochastic
from pobot.indicators.patterns import body_to_range_ratio, is_doji, is_engulfing, is_pin_bar
from pobot.indicators.trend import adx, donchian_channel, ema_slope
from pobot.indicators.volatility import atr, bollinger_bands
from pobot.indicators.volume import volume_zscore
from pobot.types import Direction

FEATURE_NAMES: list[str] = [
    "rsi_14",
    "stoch_k",
    "stoch_d",
    "macd_hist_norm",
    "ema_fast_slope",
    "ema_slow_slope",
    "close_vs_ema_fast_atr",
    "close_vs_bb_mid_atr",
    "bb_bandwidth",
    "adx_14",
    "donchian_pos",
    "volume_zscore",
    "body_to_range",
    "pin_bar_flag",
    "engulfing_flag",
    "doji_flag",
    "hour_sin",
    "hour_cos",
]


@dataclass
class FeatureRow:
    index: int
    valid: bool
    values: dict[str, float] = field(default_factory=dict)

    def as_vector(self) -> list[float]:
        return [self.values[name] for name in FEATURE_NAMES]


class FeatureBuilder:
    """Precalcula todos los indicadores una vez sobre la serie completa
    (cada uno ya es no-anticipativo) y expone filas de features por índice.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        ema_fast: int = 9,
        ema_slow: int = 21,
        bb_period: int = 20,
        atr_period: int = 14,
        adx_period: int = 14,
        donchian_period: int = 20,
        volume_period: int = 20,
    ):
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.bb_period = bb_period
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.donchian_period = donchian_period
        self.volume_period = volume_period

    def build(self, series: CandleSeries) -> list[FeatureRow]:
        candles = list(series)
        closes = series.closes()
        highs = series.highs()
        lows = series.lows()
        n = len(candles)

        rsi_vals = rsi(closes, self.rsi_period)
        stoch_k, stoch_d = stochastic(highs, lows, closes, self.stoch_period)
        macd_res = macd(closes)
        ema_fast_vals = ema(closes, self.ema_fast)
        ema_slow_vals = ema(closes, self.ema_slow)
        ema_fast_slope = ema_slope(closes, self.ema_fast, lookback=3)
        ema_slow_slope = ema_slope(closes, self.ema_slow, lookback=3)
        bb = bollinger_bands(closes, self.bb_period)
        atr_vals = atr(candles, self.atr_period)
        adx_res = adx(candles, self.adx_period)
        donchian = donchian_channel(highs, lows, self.donchian_period)
        vol_z = volume_zscore(candles, self.volume_period)
        body_ratio = body_to_range_ratio(candles)
        pin_bar = is_pin_bar(candles)
        engulfing = is_engulfing(candles)
        doji = is_doji(candles)

        rows: list[FeatureRow] = []
        for i in range(n):
            raw = {
                "rsi_14": rsi_vals[i],
                "stoch_k": stoch_k[i],
                "stoch_d": stoch_d[i],
                "macd_hist_norm": self._safe_div(macd_res.histogram[i], atr_vals[i]),
                "ema_fast_slope": ema_fast_slope[i],
                "ema_slow_slope": ema_slow_slope[i],
                "close_vs_ema_fast_atr": self._safe_div(
                    closes[i] - ema_fast_vals[i] if ema_fast_vals[i] is not None else None, atr_vals[i]
                ),
                "close_vs_bb_mid_atr": self._safe_div(
                    closes[i] - bb.middle[i] if bb.middle[i] is not None else None, atr_vals[i]
                ),
                "bb_bandwidth": bb.bandwidth[i],
                "adx_14": adx_res.adx[i],
                "donchian_pos": self._donchian_position(closes[i], donchian.upper[i], donchian.lower[i]),
                "volume_zscore": vol_z[i],
                "body_to_range": body_ratio[i],
                "pin_bar_flag": self._direction_flag(pin_bar[i]),
                "engulfing_flag": self._direction_flag(engulfing[i]),
                "doji_flag": 1.0 if doji[i] else 0.0,
                "hour_sin": math.sin(2 * math.pi * self._hour_of_day(candles[i].timestamp) / 24.0),
                "hour_cos": math.cos(2 * math.pi * self._hour_of_day(candles[i].timestamp) / 24.0),
            }
            valid = all(raw[name] is not None for name in FEATURE_NAMES)
            values = {name: (float(raw[name]) if raw[name] is not None else 0.0) for name in FEATURE_NAMES}
            rows.append(FeatureRow(index=i, valid=valid, values=values))
        return rows

    @staticmethod
    def _safe_div(numerator, denominator):
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator

    @staticmethod
    def _donchian_position(close, upper, lower):
        if upper is None or lower is None or upper == lower:
            return None
        return (close - lower) / (upper - lower)  # 0 = en el mínimo, 1 = en el máximo

    @staticmethod
    def _direction_flag(direction: Direction | None) -> float:
        if direction is Direction.CALL:
            return 1.0
        if direction is Direction.PUT:
            return -1.0
        return 0.0

    @staticmethod
    def _hour_of_day(timestamp: int) -> int:
        return (timestamp // 3600) % 24
