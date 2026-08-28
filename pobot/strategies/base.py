"""Interfaz común de estrategias y registro por nombre.

Una `Strategy` mira la serie de velas y, para la barra `i` (ya cerrada),
decide si emite una `Signal`. Solo puede usar `series[0..i]`: el backtester
y el runner en vivo confían en esa garantía para no hacer trampa.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

from pobot.data.candles import CandleSeries
from pobot.types import Direction, Signal


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def evaluate(self, series: CandleSeries, i: int) -> Optional[Signal]:
        """Evalúa la barra `i` (ya cerrada) y devuelve una señal o None.

        Implementaciones NO deben acceder a `series[j]` con `j > i`.
        """
        raise NotImplementedError

    def warmup(self) -> int:
        """Número mínimo de barras previas necesarias antes de poder evaluar."""
        return 0


class IndicatorCacheMixin:
    """Cachea arrays de indicadores precalculados sobre una serie completa.

    Los indicadores ya son no-anticipativos (el valor en `i` solo depende de
    `series[0..i]`), así que calcularlos una vez para toda la serie es
    seguro y evita recomputar ventanas en cada llamada a `evaluate`. El
    cache se invalida si cambia la identidad o el largo de la serie (cubre
    el caso de walk-forward, donde cada ventana usa un objeto distinto).
    """

    _cache_key: Optional[tuple[int, int]] = None
    _cache: object = None

    def _get_cache(self, series: CandleSeries):
        key = (id(series), len(series))
        if key != self._cache_key:
            self._cache = self._build_cache(series)
            self._cache_key = key
        return self._cache

    def _build_cache(self, series: CandleSeries):
        raise NotImplementedError


class Vote(NamedTuple):
    direction: Direction
    strength: float  # confianza en [0, 1]
    reasons: list[str]


class VotingStrategy(Strategy, ABC):
    """Estrategia que vota una dirección con una fuerza en [0,1].

    `evaluate()` convierte el voto en una `Signal` solo si la fuerza supera
    `min_confidence`, usando un número fijo de velas de expiración. Sirve de
    base tanto para estrategias individuales como para los "votantes" que
    consume `ConfluenceStrategy`.
    """

    def __init__(self, expiry_bars: int = 3, min_confidence: float = 0.55):
        self.expiry_bars = expiry_bars
        self.min_confidence = min_confidence

    @abstractmethod
    def vote(self, series: CandleSeries, i: int) -> Optional[Vote]:
        """Devuelve un voto de dirección o None si no hay condición clara.

        NO debe acceder a `series[j]` con `j > i`.
        """
        raise NotImplementedError

    def evaluate(self, series: CandleSeries, i: int) -> Optional[Signal]:
        vote = self.vote(series, i)
        if vote is None or vote.strength < self.min_confidence:
            return None
        return Signal(
            index=i,
            timestamp=series[i].timestamp,
            direction=vote.direction,
            confidence=vote.strength,
            expiry_bars=self.expiry_bars,
            strategy=self.name,
            reasons=list(vote.reasons),
        )


_REGISTRY: dict[str, type[Strategy]] = {}


def register_strategy(cls: type[Strategy]) -> type[Strategy]:
    _REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str, **kwargs) -> Strategy:
    if name not in _REGISTRY:
        raise KeyError(f"Estrategia desconocida: {name!r}. Disponibles: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_strategies() -> list[str]:
    return sorted(_REGISTRY)
