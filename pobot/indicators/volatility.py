"""Indicadores de volatilidad: ATR, Bandas de Bollinger."""

from __future__ import annotations

from typing import NamedTuple, Optional

from pobot.indicators.core import ema, rma, rolling_std, sma, true_range
from pobot.types import Candle


def atr(candles: list[Candle], period: int = 14) -> list[Optional[float]]:
    tr = true_range(candles)
    valid_start = next((i for i, v in enumerate(tr) if v is not None), len(tr))
    smoothed = rma([v for v in tr[valid_start:]], period)
    return [None] * valid_start + smoothed


class BollingerBands(NamedTuple):
    middle: list[Optional[float]]
    upper: list[Optional[float]]
    lower: list[Optional[float]]
    bandwidth: list[Optional[float]]  # (upper - lower) / middle


def bollinger_bands(closes: list[float], period: int = 20, num_std: float = 2.0) -> BollingerBands:
    middle = sma(closes, period)
    std = rolling_std(closes, period)
    upper: list[Optional[float]] = [None] * len(closes)
    lower: list[Optional[float]] = [None] * len(closes)
    bandwidth: list[Optional[float]] = [None] * len(closes)
    for i in range(len(closes)):
        if middle[i] is None or std[i] is None:
            continue
        upper[i] = middle[i] + num_std * std[i]
        lower[i] = middle[i] - num_std * std[i]
        bandwidth[i] = 0.0 if middle[i] == 0 else (upper[i] - lower[i]) / abs(middle[i])
    return BollingerBands(middle, upper, lower, bandwidth)


def keltner_channel(
    candles: list[Candle], ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0
) -> tuple[list[Optional[float]], list[Optional[float]], list[Optional[float]]]:
    closes = [c.close for c in candles]
    middle = ema(closes, ema_period)
    atr_vals = atr(candles, atr_period)
    upper: list[Optional[float]] = [None] * len(candles)
    lower: list[Optional[float]] = [None] * len(candles)
    for i in range(len(candles)):
        if middle[i] is None or atr_vals[i] is None:
            continue
        upper[i] = middle[i] + multiplier * atr_vals[i]
        lower[i] = middle[i] - multiplier * atr_vals[i]
    return middle, upper, lower
