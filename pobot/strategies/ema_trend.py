"""Seguimiento de tendencia por cruce/alineación de EMAs.

Vota CALL cuando la EMA rápida está por encima de la lenta y ambas tienen
pendiente positiva reciente (tendencia alcista confirmada); PUT en el caso
simétrico. No vota en mercados laterales o con EMAs en pendientes opuestas.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

from pobot.data.candles import CandleSeries
from pobot.indicators.core import ema
from pobot.indicators.trend import ema_slope
from pobot.strategies.base import IndicatorCacheMixin, Vote, VotingStrategy, register_strategy
from pobot.types import Direction


class _EMATrendCache(NamedTuple):
    fast: list
    slow: list
    fast_slope: list
    slow_slope: list


@register_strategy
class EMATrendStrategy(IndicatorCacheMixin, VotingStrategy):
    name = "ema_trend"

    def __init__(
        self,
        fast_period: int = 9,
        slow_period: int = 21,
        slope_lookback: int = 3,
        expiry_bars: int = 3,
        min_confidence: float = 0.55,
    ):
        super().__init__(expiry_bars=expiry_bars, min_confidence=min_confidence)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.slope_lookback = slope_lookback

    def warmup(self) -> int:
        return self.slow_period + self.slope_lookback + 1

    def _build_cache(self, series: CandleSeries) -> _EMATrendCache:
        closes = series.closes()
        return _EMATrendCache(
            fast=ema(closes, self.fast_period),
            slow=ema(closes, self.slow_period),
            fast_slope=ema_slope(closes, self.fast_period, self.slope_lookback),
            slow_slope=ema_slope(closes, self.slow_period, self.slope_lookback),
        )

    def vote(self, series: CandleSeries, i: int) -> Optional[Vote]:
        cache = self._get_cache(series)
        fast, slow = cache.fast[i], cache.slow[i]
        fslope, sslope = cache.fast_slope[i], cache.slow_slope[i]
        if None in (fast, slow, fslope, sslope) or slow == 0:
            return None

        separation = abs(fast - slow) / abs(slow)
        strength = 0.5 + 0.5 * min(1.0, separation * 20)  # separación típica ~2-3% -> fuerza cerca de 1

        if fast > slow and fslope > 0 and sslope > 0:
            return Vote(
                Direction.CALL, strength,
                [f"EMA{self.fast_period} > EMA{self.slow_period} con pendientes alcistas"],
            )
        if fast < slow and fslope < 0 and sslope < 0:
            return Vote(
                Direction.PUT, strength,
                [f"EMA{self.fast_period} < EMA{self.slow_period} con pendientes bajistas"],
            )
        return None
