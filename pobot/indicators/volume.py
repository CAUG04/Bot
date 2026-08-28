"""Indicadores de volumen: OBV, VWAP de sesión, z-score de volumen."""

from __future__ import annotations

from typing import Optional

from pobot.types import Candle


def obv(candles: list[Candle]) -> list[float]:
    """On-Balance Volume: acumula volumen con signo según la dirección de la vela."""
    out: list[float] = [0.0] * len(candles)
    running = 0.0
    for i, c in enumerate(candles):
        if i > 0:
            if c.close > candles[i - 1].close:
                running += c.volume
            elif c.close < candles[i - 1].close:
                running -= c.volume
        out[i] = running
    return out


def session_vwap(candles: list[Candle], session_seconds: int = 86400) -> list[Optional[float]]:
    """VWAP que se reinicia cada `session_seconds` (por defecto, sesiones diarias UTC)."""
    out: list[Optional[float]] = [None] * len(candles)
    if not candles:
        return out
    session_start = (candles[0].timestamp // session_seconds) * session_seconds
    cum_pv = 0.0
    cum_v = 0.0
    for i, c in enumerate(candles):
        current_session = (c.timestamp // session_seconds) * session_seconds
        if current_session != session_start:
            session_start = current_session
            cum_pv = 0.0
            cum_v = 0.0
        typical_price = (c.high + c.low + c.close) / 3.0
        cum_pv += typical_price * c.volume
        cum_v += c.volume
        out[i] = (cum_pv / cum_v) if cum_v > 0 else typical_price
    return out


def volume_zscore(candles: list[Candle], period: int = 20) -> list[Optional[float]]:
    from pobot.indicators.core import rolling_std, sma

    volumes = [c.volume for c in candles]
    mean = sma(volumes, period)
    std = rolling_std(volumes, period)
    out: list[Optional[float]] = [None] * len(candles)
    for i in range(len(candles)):
        if mean[i] is None or std[i] is None or std[i] == 0:
            continue
        out[i] = (volumes[i] - mean[i]) / std[i]
    return out
