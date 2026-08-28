"""Importar el paquete registra automáticamente todas las estrategias
concretas en `pobot.strategies.base` (vía el decorador `@register_strategy`),
para que `get_strategy(nombre)` funcione sin que el caller tenga que saber
qué módulo concreto define cada nombre.
"""

from pobot.strategies import (  # noqa: F401
    bollinger_bounce,
    confluence,
    ema_trend,
    macd_momentum,
    ml_strategy,
    pattern_sr,
    rsi_reversal,
)
