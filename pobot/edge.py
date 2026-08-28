"""Matemática de ventaja (edge) para opciones binarias.

En una opción binaria de payout `p` (ej. 0.92 = 92%): acertar paga `p * stake`,
fallar cuesta `stake`. Con probabilidad de acierto `w`:

    EV = w * p - (1 - w) = w * (1 + p) - 1

El winrate de equilibrio (EV = 0) es:

    w_be = 1 / (1 + p)

Todo el sistema se evalúa contra este umbral. Una racha ganadora no prueba
ventaja: hay que mirar el límite inferior del intervalo de confianza.
"""

from __future__ import annotations

import math


def breakeven_winrate(payout: float) -> float:
    """Winrate mínimo para EV >= 0 dado un payout (ej. 0.92)."""
    if payout <= 0:
        raise ValueError("payout debe ser > 0")
    return 1.0 / (1.0 + payout)


def expected_value(winrate: float, payout: float, stake: float = 1.0) -> float:
    """EV monetario esperado por operación con el stake dado."""
    if not 0.0 <= winrate <= 1.0:
        raise ValueError("winrate debe estar en [0, 1]")
    return stake * (winrate * (1.0 + payout) - 1.0)


def roi(winrate: float, payout: float) -> float:
    """Retorno esperado por unidad apostada (ROI), equivalente a expected_value(stake=1)."""
    return expected_value(winrate, payout, stake=1.0)


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza de Wilson para una proporción binomial.

    Más fiable que el intervalo normal con muestras pequeñas o winrates
    cercanos a 0/1, que es justo el régimen en el que suelen operar los bots
    de señales con pocos backtests.

    Devuelve (límite_inferior, límite_superior) en [0, 1].
    """
    if n <= 0:
        raise ValueError("n debe ser > 0")
    if not 0 <= wins <= n:
        raise ValueError("wins debe estar en [0, n]")

    p_hat = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p_hat + z2 / (2 * n)
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * n)) / n)

    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return max(0.0, lower), min(1.0, upper)


def has_demonstrated_edge(wins: int, n: int, payout: float, z: float = 1.96) -> bool:
    """True solo si el límite inferior de Wilson supera el winrate de equilibrio.

    Este es el criterio de aceptación de cualquier estrategia o modelo en este
    proyecto. Un winrate observado alto con pocas muestras NO basta.
    """
    if n == 0:
        return False
    lower, _ = wilson_interval(wins, n, z)
    return lower > breakeven_winrate(payout)


def kelly_fraction(winrate: float, payout: float, cap: float = 1.0) -> float:
    """Fracción de Kelly para una apuesta binaria (b = payout, p = winrate).

    f* = p - (1 - p) / b

    Se recorta a [0, cap] porque Kelly completo es demasiado agresivo para
    binarias reales (los winrates estimados llevan error de muestreo). Un
    valor negativo significa que no hay ventaja y no se debe apostar nada.
    """
    if payout <= 0:
        raise ValueError("payout debe ser > 0")
    f = winrate - (1.0 - winrate) / payout
    return max(0.0, min(cap, f))
