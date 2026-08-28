"""Estrategia de confluencia: combina varias `VotingStrategy` por votación
ponderada y solo emite señal cuando hay acuerdo suficiente.

Esto es lo que responde "bajo qué condiciones entrar": no basta con que un
indicador dispare, se exige que el peso neto a favor de una dirección supere
`min_net_strength` y que al menos `min_voters` sub-estrategias voten esa
misma dirección.
"""

from __future__ import annotations

from typing import Optional

from pobot.data.candles import CandleSeries
from pobot.strategies.base import Strategy, VotingStrategy, register_strategy
from pobot.strategies.bollinger_bounce import BollingerBounceStrategy
from pobot.strategies.ema_trend import EMATrendStrategy
from pobot.strategies.macd_momentum import MACDMomentumStrategy
from pobot.strategies.pattern_sr import PatternSRStrategy
from pobot.strategies.rsi_reversal import RSIReversalStrategy
from pobot.types import Direction, Signal

DEFAULT_VOTERS: list[tuple[VotingStrategy, float]] = []  # se construye perezosamente, ver _default_voters()


def _default_voters() -> list[tuple[VotingStrategy, float]]:
    return [
        (RSIReversalStrategy(min_confidence=0.0), 1.0),
        (EMATrendStrategy(min_confidence=0.0), 1.0),
        (BollingerBounceStrategy(min_confidence=0.0), 1.0),
        (MACDMomentumStrategy(min_confidence=0.0), 1.0),
        (PatternSRStrategy(min_confidence=0.0), 1.2),
    ]


@register_strategy
class ConfluenceStrategy(Strategy):
    name = "confluence"

    def __init__(
        self,
        voters: Optional[list[tuple[VotingStrategy, float]]] = None,
        min_voters: int = 2,
        min_net_strength: float = 0.55,
        expiry_bars: int = 3,
    ):
        self.voters = voters if voters is not None else _default_voters()
        self.min_voters = min_voters
        self.min_net_strength = min_net_strength
        self.expiry_bars = expiry_bars

    def warmup(self) -> int:
        return max((v.warmup() for v, _ in self.voters), default=0)

    def evaluate(self, series: CandleSeries, i: int) -> Optional[Signal]:
        call_weight = 0.0
        put_weight = 0.0
        call_votes = 0
        put_votes = 0
        reasons: list[str] = []

        for voter, weight in self.voters:
            vote = voter.vote(series, i)
            if vote is None:
                continue
            if vote.direction is Direction.CALL:
                call_weight += weight * vote.strength
                call_votes += 1
            else:
                put_weight += weight * vote.strength
                put_votes += 1
            reasons.extend(vote.reasons)

        total_weight = call_weight + put_weight
        if total_weight == 0:
            return None

        if call_weight > put_weight:
            direction, votes, side_weight = Direction.CALL, call_votes, call_weight
        else:
            direction, votes, side_weight = Direction.PUT, put_votes, put_weight

        net_strength = side_weight / total_weight
        if votes < self.min_voters or net_strength < self.min_net_strength:
            return None

        return Signal(
            index=i,
            timestamp=series[i].timestamp,
            direction=direction,
            confidence=net_strength,
            expiry_bars=self.expiry_bars,
            strategy=self.name,
            reasons=reasons,
        )
