"""Indicadores de momentum: RSI, MACD, Estocástico, ROC."""

from __future__ import annotations

from typing import NamedTuple, Optional

from pobot.indicators.core import ema, rma, rolling_max, rolling_min


def rsi(closes: list[float], period: int = 14) -> list[Optional[float]]:
    if len(closes) < 2:
        return [None] * len(closes)
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = rma(gains, period)
    avg_loss = rma(losses, period)
    out: list[Optional[float]] = [None] * len(closes)
    for i in range(len(closes)):
        if avg_gain[i] is None or avg_loss[i] is None:
            continue
        if avg_loss[i] == 0:
            out[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


class MACDResult(NamedTuple):
    macd: list[Optional[float]]
    signal: list[Optional[float]]
    histogram: list[Optional[float]]


def macd(
    closes: list[float], fast: int = 12, slow: int = 26, signal_period: int = 9
) -> MACDResult:
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line: list[Optional[float]] = [
        (f - s) if (f is not None and s is not None) else None
        for f, s in zip(ema_fast, ema_slow)
    ]
    # ema() sobre la línea MACD debe ignorar los None iniciales; se calcula
    # sobre la subserie válida y se reinserta alineada.
    valid_start = next((i for i, v in enumerate(macd_line) if v is not None), len(macd_line))
    signal_valid = ema([v for v in macd_line[valid_start:]], signal_period)
    signal_line: list[Optional[float]] = [None] * valid_start + signal_valid
    histogram: list[Optional[float]] = [
        (m - s) if (m is not None and s is not None) else None
        for m, s in zip(macd_line, signal_line)
    ]
    return MACDResult(macd_line, signal_line, histogram)


def stochastic(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14, smooth_k: int = 3
) -> tuple[list[Optional[float]], list[Optional[float]]]:
    highest = rolling_max(highs, period)
    lowest = rolling_min(lows, period)
    raw_k: list[Optional[float]] = [None] * len(closes)
    for i in range(len(closes)):
        if highest[i] is None or lowest[i] is None:
            continue
        rng = highest[i] - lowest[i]
        raw_k[i] = 50.0 if rng == 0 else 100.0 * (closes[i] - lowest[i]) / rng
    valid_start = next((i for i, v in enumerate(raw_k) if v is not None), len(raw_k))
    from pobot.indicators.core import sma

    k_smoothed_valid = sma([v for v in raw_k[valid_start:]], smooth_k)
    k = [None] * valid_start + k_smoothed_valid
    k_valid_start = next((i for i, v in enumerate(k) if v is not None), len(k))
    d_valid = sma([v for v in k[k_valid_start:]], smooth_k)
    d = [None] * k_valid_start + d_valid
    return k, d


def roc(closes: list[float], period: int = 10) -> list[Optional[float]]:
    """Rate of change porcentual respecto a `period` barras atrás."""
    out: list[Optional[float]] = [None] * len(closes)
    for i in range(period, len(closes)):
        base = closes[i - period]
        out[i] = 0.0 if base == 0 else 100.0 * (closes[i] - base) / base
    return out
