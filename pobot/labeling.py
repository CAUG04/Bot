"""Etiquetado de dirección para entrenar/evaluar modelos de opciones binarias.

Regla de negocio (como Pocket Option): la operación abierta al cierre de la
barra `t` entra al **open de la barra t+1** y se liquida contra el
**close de la barra t+horizon**. La etiqueta de la fila `t` es CALL si ese
cierre queda por encima del precio de entrada, PUT si queda por debajo, y
`None` (empate) si coincide exactamente.

Las últimas `horizon` barras de la serie no tienen etiqueta posible (no hay
futuro suficiente) y se dejan explícitamente como `None`: nunca se rellenan
ni se extrapolan, porque eso introduciría look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pobot.data.candles import CandleSeries
from pobot.types import Direction


@dataclass
class LabelSet:
    horizon: int
    tie_policy: str  # "refund" | "loss"
    entry_price: list[Optional[float]]
    exit_price: list[Optional[float]]
    direction: list[Optional[Direction]]  # None = sin etiqueta (fin de serie) o empate en modo refund

    def __len__(self) -> int:
        return len(self.direction)


def make_labels(series: CandleSeries, horizon: int, tie_policy: str = "refund") -> LabelSet:
    if horizon < 1:
        raise ValueError("horizon debe ser >= 1")
    if tie_policy not in ("refund", "loss"):
        raise ValueError("tie_policy debe ser 'refund' o 'loss'")

    n = len(series)
    entry_price: list[Optional[float]] = [None] * n
    exit_price: list[Optional[float]] = [None] * n
    direction: list[Optional[Direction]] = [None] * n

    for t in range(n):
        entry_idx = t + 1
        exit_idx = t + horizon
        if exit_idx >= n:
            continue  # no hay futuro suficiente: sin etiqueta, punto final
        entry = series[entry_idx].open
        exit_ = series[exit_idx].close
        entry_price[t] = entry
        exit_price[t] = exit_

        if exit_ > entry:
            direction[t] = Direction.CALL
        elif exit_ < entry:
            direction[t] = Direction.PUT
        else:
            direction[t] = None if tie_policy == "refund" else Direction.PUT

    return LabelSet(horizon=horizon, tie_policy=tie_policy, entry_price=entry_price, exit_price=exit_price, direction=direction)
