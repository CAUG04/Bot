"""Indicadores de tendencia: ADX, canal de Donchian, pendiente de EMA."""

from __future__ import annotations

from typing import NamedTuple, Optional

from pobot.indicators.core import ema, rma, rolling_max, rolling_min, true_range
from pobot.types import Candle


class ADXResult(NamedTuple):
    adx: list[Optional[float]]
    plus_di: list[Optional[float]]
    minus_di: list[Optional[float]]


def adx(candles: list[Candle], period: int = 14) -> ADXResult:
    n = len(candles)
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    for i in range(1, n):
        up_move = candles[i].high - candles[i - 1].high
        down_move = candles[i - 1].low - candles[i].low
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

    tr = true_range(candles)
    valid_start = next((i for i, v in enumerate(tr) if v is not None), n)

    smoothed_tr = rma([v for v in tr[valid_start:]], period)
    smoothed_plus = rma(plus_dm[valid_start:], period)
    smoothed_minus = rma(minus_dm[valid_start:], period)

    plus_di: list[Optional[float]] = [None] * valid_start
    minus_di: list[Optional[float]] = [None] * valid_start
    dx: list[Optional[float]] = [None] * valid_start

    for str_, spd, smd in zip(smoothed_tr, smoothed_plus, smoothed_minus):
        if str_ is None or str_ == 0:
            plus_di.append(None)
            minus_di.append(None)
            dx.append(None)
            continue
        pdi = 100.0 * spd / str_
        mdi = 100.0 * smd / str_
        plus_di.append(pdi)
        minus_di.append(mdi)
        denom = pdi + mdi
        dx.append(0.0 if denom == 0 else 100.0 * abs(pdi - mdi) / denom)

    dx_valid_start = next((i for i, v in enumerate(dx) if v is not None), n)
    adx_valid = rma([v for v in dx[dx_valid_start:]], period)
    adx_line = [None] * dx_valid_start + adx_valid

    return ADXResult(adx_line, plus_di, minus_di)


class DonchianChannel(NamedTuple):
    upper: list[Optional[float]]
    lower: list[Optional[float]]
    middle: list[Optional[float]]


def donchian_channel(highs: list[float], lows: list[float], period: int = 20) -> DonchianChannel:
    upper = rolling_max(highs, period)
    lower = rolling_min(lows, period)
    middle = [
        (u + l) / 2.0 if (u is not None and l is not None) else None
        for u, l in zip(upper, lower)
    ]
    return DonchianChannel(upper, lower, middle)


def ema_slope(closes: list[float], period: int = 20, lookback: int = 3) -> list[Optional[float]]:
    """Pendiente de la EMA normalizada: (ema[i] - ema[i-lookback]) / ema[i-lookback].

    Positiva = tendencia alcista reciente; negativa = bajista. Se normaliza
    para que sea comparable entre activos con precios de magnitudes distintas.
    """
    values = ema(closes, period)
    out: list[Optional[float]] = [None] * len(closes)
    for i in range(len(closes)):
        if i < lookback:
            continue
        prev = values[i - lookback]
        curr = values[i]
        if prev is None or curr is None or prev == 0:
            continue
        out[i] = (curr - prev) / abs(prev)
    return out
