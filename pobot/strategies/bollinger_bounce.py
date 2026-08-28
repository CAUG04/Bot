"""Rebote en Bandas de Bollinger: precio tocando/rompiendo la banda inferior
vota CALL (reversión al alza), tocando la superior vota PUT.
"""

from __future__ import annotations

from typing import Optional

from pobot.data.candles import CandleSeries
from pobot.indicators.volatility import bollinger_bands
from pobot.strategies.base import IndicatorCacheMixin, Vote, VotingStrategy, register_strategy
from pobot.types import Direction


@register_strategy
class BollingerBounceStrategy(IndicatorCacheMixin, VotingStrategy):
    name = "bollinger_bounce"

    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
        expiry_bars: int = 2,
        min_confidence: float = 0.55,
    ):
        super().__init__(expiry_bars=expiry_bars, min_confidence=min_confidence)
        self.period = period
        self.num_std = num_std

    def warmup(self) -> int:
        return self.period + 1

    def _build_cache(self, series: CandleSeries):
        return bollinger_bands(series.closes(), self.period, self.num_std)

    def vote(self, series: CandleSeries, i: int) -> Optional[Vote]:
        bands = self._get_cache(series)
        upper, lower, middle = bands.upper[i], bands.lower[i], bands.middle[i]
        if None in (upper, lower, middle) or upper == lower:
            return None

        close = series[i].close
        half_width = (upper - lower) / 2.0

        if close <= lower:
            penetration = (lower - close) / half_width if half_width > 0 else 0.0
            strength = 0.5 + 0.5 * min(1.0, 0.3 + penetration)
            return Vote(Direction.CALL, strength, ["Precio en/bajo banda inferior de Bollinger"])
        if close >= upper:
            penetration = (close - upper) / half_width if half_width > 0 else 0.0
            strength = 0.5 + 0.5 * min(1.0, 0.3 + penetration)
            return Vote(Direction.PUT, strength, ["Precio en/sobre banda superior de Bollinger"])
        return None
