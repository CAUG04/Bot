"""Bloques básicos de indicadores: medias móviles y volatilidad rodante.

Todas las funciones son O(n) totales (O(1) amortizado por barra, sin
recalcular ventanas completas en cada paso) y devuelven una lista alineada
1:1 con la entrada, con `None` en las posiciones donde aún no hay historial
suficiente. Ninguna función mira valores futuros: el valor en el índice `i`
depende solo de `data[0..i]`.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from pobot.types import Candle

NA = None  # alias legible para "no disponible aún"


def sma(values: list[float], period: int) -> list[Optional[float]]:
    if period < 1:
        raise ValueError("period debe ser >= 1")
    out: list[Optional[float]] = [None] * len(values)
    running_sum = 0.0
    for i, v in enumerate(values):
        running_sum += v
        if i >= period:
            running_sum -= values[i - period]
        if i >= period - 1:
            out[i] = running_sum / period
    return out


def ema(values: list[float], period: int) -> list[Optional[float]]:
    if period < 1:
        raise ValueError("period debe ser >= 1")
    out: list[Optional[float]] = [None] * len(values)
    alpha = 2.0 / (period + 1)
    prev: Optional[float] = None
    seed_sum = 0.0
    for i, v in enumerate(values):
        if i < period - 1:
            seed_sum += v
            continue
        if i == period - 1:
            seed_sum += v
            prev = seed_sum / period
        else:
            prev = alpha * v + (1 - alpha) * prev
        out[i] = prev
    return out


def rma(values: list[float], period: int) -> list[Optional[float]]:
    """Media móvil de Wilder (RMA), usada en RSI/ATR/ADX clásicos."""
    if period < 1:
        raise ValueError("period debe ser >= 1")
    out: list[Optional[float]] = [None] * len(values)
    prev: Optional[float] = None
    seed_sum = 0.0
    for i, v in enumerate(values):
        if i < period - 1:
            seed_sum += v
            continue
        if i == period - 1:
            seed_sum += v
            prev = seed_sum / period
        else:
            prev = (prev * (period - 1) + v) / period
        out[i] = prev
    return out


def rolling_std(values: list[float], period: int) -> list[Optional[float]]:
    """Desviación típica poblacional en una ventana rodante de tamaño `period`."""
    if period < 2:
        raise ValueError("period debe ser >= 2")
    out: list[Optional[float]] = [None] * len(values)
    window: list[float] = []
    running_sum = 0.0
    running_sq = 0.0
    for i, v in enumerate(values):
        window.append(v)
        running_sum += v
        running_sq += v * v
        if len(window) > period:
            old = window.pop(0)
            running_sum -= old
            running_sq -= old * old
        if len(window) == period:
            mean = running_sum / period
            variance = max(0.0, running_sq / period - mean * mean)
            out[i] = variance**0.5
    return out


def true_range(candles: list[Candle]) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(candles)
    for i, c in enumerate(candles):
        if i == 0:
            out[i] = c.high - c.low
        else:
            prev_close = candles[i - 1].close
            out[i] = max(
                c.high - c.low,
                abs(c.high - prev_close),
                abs(c.low - prev_close),
            )
    return out


def _rolling_extreme(values: list[float], period: int, keep_larger: bool) -> list[Optional[float]]:
    """Mín/máx rodante en O(n) total via deque monótono (amortizado O(1) por barra)."""
    out: list[Optional[float]] = [None] * len(values)
    dq: deque[int] = deque()  # índices, valores en orden monótono
    for i, v in enumerate(values):
        while dq and (values[dq[-1]] <= v if keep_larger else values[dq[-1]] >= v):
            dq.pop()
        dq.append(i)
        if dq[0] <= i - period:
            dq.popleft()
        if i >= period - 1:
            out[i] = values[dq[0]]
    return out


def rolling_min(values: list[float], period: int) -> list[Optional[float]]:
    return _rolling_extreme(values, period, keep_larger=False)


def rolling_max(values: list[float], period: int) -> list[Optional[float]]:
    return _rolling_extreme(values, period, keep_larger=True)
