"""Tipos de datos fundamentales compartidos por todo el paquete."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Direction(str, Enum):
    """Dirección de una operación binaria."""

    CALL = "CALL"  # apuesta a que sube
    PUT = "PUT"  # apuesta a que baja

    def opposite(self) -> "Direction":
        return Direction.PUT if self is Direction.CALL else Direction.CALL


class TradeResult(str, Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    TIE = "TIE"  # empate: Pocket Option suele devolver el stake


@dataclass(frozen=True)
class Candle:
    """Una vela OHLCV. `timestamp` es epoch en segundos, UTC, cierre de la vela."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def is_bullish(self) -> bool:
        return self.close > self.open

    def is_bearish(self) -> bool:
        return self.close < self.open

    def body(self) -> float:
        return abs(self.close - self.open)

    def range(self) -> float:
        return self.high - self.low

    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low


@dataclass
class Signal:
    """Señal emitida por una estrategia para la barra `index`."""

    index: int
    timestamp: int
    direction: Direction
    confidence: float  # probabilidad estimada en [0, 1] de que direction acierte
    expiry_bars: int
    strategy: str
    reasons: list[str] = field(default_factory=list)
    entry_price: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "direction": self.direction.value,
            "confidence": round(self.confidence, 4),
            "expiry_bars": self.expiry_bars,
            "strategy": self.strategy,
            "reasons": list(self.reasons),
            "entry_price": self.entry_price,
        }


@dataclass
class Trade:
    """Una operación simulada o real, ya cerrada."""

    entry_index: int
    entry_timestamp: int
    exit_index: int
    exit_timestamp: int
    direction: Direction
    entry_price: float
    exit_price: float
    expiry_bars: int
    stake: float
    payout: float  # ej. 0.92 significa 92% de ganancia sobre el stake
    strategy: str
    confidence: float = 0.0
    forced_result: Optional[TradeResult] = None  # usado por tie_policy="loss" sin falsear el precio

    @property
    def result(self) -> TradeResult:
        if self.forced_result is not None:
            return self.forced_result
        if self.exit_price == self.entry_price:
            return TradeResult.TIE
        went_up = self.exit_price > self.entry_price
        won = went_up if self.direction is Direction.CALL else not went_up
        return TradeResult.WIN if won else TradeResult.LOSS

    @property
    def pnl(self) -> float:
        result = self.result
        if result is TradeResult.WIN:
            return self.stake * self.payout
        if result is TradeResult.LOSS:
            return -self.stake
        return 0.0
