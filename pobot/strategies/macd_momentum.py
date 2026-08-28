"""Momentum por cruce de MACD: histograma cruzando a positivo vota CALL,
cruzando a negativo vota PUT. Requiere el cruce (no solo el signo) para
evitar votar en cada barra de una tendencia ya establecida.
"""

from __future__ import annotations

from typing import Optional

from pobot.data.candles import CandleSeries
from pobot.indicators.momentum import macd
from pobot.strategies.base import IndicatorCacheMixin, Vote, VotingStrategy, register_strategy
from pobot.types import Direction


@register_strategy
class MACDMomentumStrategy(IndicatorCacheMixin, VotingStrategy):
    name = "macd_momentum"

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal_period: int = 9,
        expiry_bars: int = 3,
        min_confidence: float = 0.55,
    ):
        super().__init__(expiry_bars=expiry_bars, min_confidence=min_confidence)
        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period

    def warmup(self) -> int:
        return self.slow + self.signal_period + 1

    def _build_cache(self, series: CandleSeries):
        return macd(series.closes(), self.fast, self.slow, self.signal_period)

    def vote(self, series: CandleSeries, i: int) -> Optional[Vote]:
        if i < 1:
            return None
        cache = self._get_cache(series)
        hist, hist_prev = cache.histogram[i], cache.histogram[i - 1]
        if hist is None or hist_prev is None:
            return None

        atr_scale = abs(hist) + abs(hist_prev) + 1e-9
        strength = 0.5 + 0.5 * min(1.0, abs(hist) / atr_scale * 2)

        if hist_prev <= 0 < hist:
            return Vote(Direction.CALL, strength, ["Cruce alcista del histograma MACD"])
        if hist_prev >= 0 > hist:
            return Vote(Direction.PUT, strength, ["Cruce bajista del histograma MACD"])
        return None
