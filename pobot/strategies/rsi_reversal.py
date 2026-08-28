"""Reversión por RSI: sobreventa -> voto CALL, sobrecompra -> voto PUT.

La fuerza del voto crece cuanto más extremo está el RSI respecto al umbral.
"""

from __future__ import annotations

from typing import Optional

from pobot.data.candles import CandleSeries
from pobot.indicators.momentum import rsi
from pobot.strategies.base import IndicatorCacheMixin, Vote, VotingStrategy, register_strategy
from pobot.types import Direction


@register_strategy
class RSIReversalStrategy(IndicatorCacheMixin, VotingStrategy):
    name = "rsi_reversal"

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        expiry_bars: int = 3,
        min_confidence: float = 0.55,
    ):
        super().__init__(expiry_bars=expiry_bars, min_confidence=min_confidence)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def warmup(self) -> int:
        return self.period + 1

    def _build_cache(self, series: CandleSeries):
        return rsi(series.closes(), self.period)

    def vote(self, series: CandleSeries, i: int) -> Optional[Vote]:
        rsi_values = self._get_cache(series)
        value = rsi_values[i]
        if value is None:
            return None
        if value <= self.oversold:
            strength = 0.5 + 0.5 * min(1.0, (self.oversold - value) / self.oversold)
            return Vote(Direction.CALL, strength, [f"RSI({self.period})={value:.1f} <= {self.oversold} (sobreventa)"])
        if value >= self.overbought:
            span = max(1e-9, 100.0 - self.overbought)
            strength = 0.5 + 0.5 * min(1.0, (value - self.overbought) / span)
            return Vote(Direction.PUT, strength, [f"RSI({self.period})={value:.1f} >= {self.overbought} (sobrecompra)"])
        return None
