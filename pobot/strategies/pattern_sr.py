"""Patrón de vela confirmado en soporte/resistencia (canal de Donchian).

Un pin bar o envolvente alcista cerca del mínimo reciente vota CALL; el caso
bajista simétrico cerca del máximo reciente vota PUT. Exigir la confluencia
patrón + nivel evita operar el patrón "en el vacío", lejos de cualquier
soporte/resistencia relevante.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

from pobot.data.candles import CandleSeries
from pobot.indicators.patterns import is_engulfing, is_pin_bar
from pobot.indicators.trend import donchian_channel
from pobot.strategies.base import IndicatorCacheMixin, Vote, VotingStrategy, register_strategy
from pobot.types import Direction


class _PatternSRCache(NamedTuple):
    donchian_upper: list
    donchian_lower: list
    pin_bar: list
    engulfing: list


@register_strategy
class PatternSRStrategy(IndicatorCacheMixin, VotingStrategy):
    name = "pattern_sr"

    def __init__(
        self,
        donchian_period: int = 20,
        proximity_pct: float = 0.15,
        expiry_bars: int = 2,
        min_confidence: float = 0.55,
    ):
        super().__init__(expiry_bars=expiry_bars, min_confidence=min_confidence)
        self.donchian_period = donchian_period
        # proximity_pct: fracción del ancho del canal que cuenta como "cerca" del borde
        self.proximity_pct = proximity_pct

    def warmup(self) -> int:
        return self.donchian_period + 2

    def _build_cache(self, series: CandleSeries) -> _PatternSRCache:
        candles = list(series)
        donchian = donchian_channel(series.highs(), series.lows(), self.donchian_period)
        return _PatternSRCache(
            donchian_upper=donchian.upper,
            donchian_lower=donchian.lower,
            pin_bar=is_pin_bar(candles),
            engulfing=is_engulfing(candles),
        )

    def vote(self, series: CandleSeries, i: int) -> Optional[Vote]:
        if i < 1:
            return None
        cache = self._get_cache(series)
        # El nivel de soporte/resistencia se mide con el canal HASTA LA BARRA
        # ANTERIOR: si se incluyera la barra actual, la propia mecha del pin
        # bar movería el canal y el precio nunca parecería "cerca del borde".
        upper, lower = cache.donchian_upper[i - 1], cache.donchian_lower[i - 1]
        if upper is None or lower is None or upper == lower:
            return None

        close = series[i].close
        width = upper - lower
        near_lower = (close - lower) <= self.proximity_pct * width
        near_upper = (upper - close) <= self.proximity_pct * width

        pattern = cache.pin_bar[i] or cache.engulfing[i]
        if pattern is Direction.CALL and near_lower:
            return Vote(Direction.CALL, 0.7, ["Patrón alcista cerca del soporte de Donchian"])
        if pattern is Direction.PUT and near_upper:
            return Vote(Direction.PUT, 0.7, ["Patrón bajista cerca de la resistencia de Donchian"])
        return None
