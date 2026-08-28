"""Patrones de vela: envolvente, pin bar, doji, ratio cuerpo/mecha.

Cada función devuelve una lista alineada de flags/valores para la barra `i`,
calculados solo con `candles[i]` y, cuando aplica, `candles[i-1]` (nunca con
barras futuras).
"""

from __future__ import annotations

from typing import Optional

from pobot.types import Candle, Direction


def body_to_range_ratio(candles: list[Candle]) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(candles)
    for i, c in enumerate(candles):
        rng = c.range()
        out[i] = 0.0 if rng == 0 else c.body() / rng
    return out


def is_doji(candles: list[Candle], threshold: float = 0.1) -> list[bool]:
    """Vela con cuerpo pequeño respecto a su rango total (indecisión)."""
    ratios = body_to_range_ratio(candles)
    return [(r is not None and r <= threshold) for r in ratios]


def is_pin_bar(candles: list[Candle], wick_ratio: float = 2.0) -> list[Optional[Direction]]:
    """Pin bar: una mecha domina el rango, señal de rechazo de precio.

    Mecha inferior larga -> posible reversión alcista (Direction.CALL).
    Mecha superior larga -> posible reversión bajista (Direction.PUT).
    """
    out: list[Optional[Direction]] = [None] * len(candles)
    for i, c in enumerate(candles):
        body = c.body()
        if body == 0:
            continue
        lower = c.lower_wick()
        upper = c.upper_wick()
        if lower >= wick_ratio * body and lower > upper:
            out[i] = Direction.CALL
        elif upper >= wick_ratio * body and upper > lower:
            out[i] = Direction.PUT
    return out


def is_engulfing(candles: list[Candle]) -> list[Optional[Direction]]:
    """Envolvente alcista/bajista: el cuerpo de la vela actual cubre el de la anterior."""
    out: list[Optional[Direction]] = [None] * len(candles)
    for i in range(1, len(candles)):
        prev, curr = candles[i - 1], candles[i]
        if prev.body() == 0:
            continue
        curr_covers_prev = (
            max(curr.open, curr.close) >= max(prev.open, prev.close)
            and min(curr.open, curr.close) <= min(prev.open, prev.close)
        )
        if not curr_covers_prev:
            continue
        if prev.is_bearish() and curr.is_bullish():
            out[i] = Direction.CALL
        elif prev.is_bullish() and curr.is_bearish():
            out[i] = Direction.PUT
    return out
