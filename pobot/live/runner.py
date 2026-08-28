"""Runner de señales en vivo: sondea velas, espera el cierre confirmado de la
barra, evalúa la estrategia (ya validada en backtest) y notifica.

`fetch_candles` se inyecta como función (no se acopla a Binance directamente)
para poder testear el runner con datos fijos y para poder cambiar de fuente
de datos sin tocar esta clase.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from pobot.data.candles import CandleSeries
from pobot.live.notifier import format_signal_message
from pobot.strategies.base import Strategy
from pobot.types import Signal


@dataclass
class LiveRunnerConfig:
    symbol: str
    interval: str
    payout: float
    poll_seconds: float = 5.0
    allowed_hours: Optional[list[int]] = None


class LiveRunner:
    def __init__(
        self,
        strategy: Strategy,
        notifier,
        fetch_candles: Callable[[], CandleSeries],
        config: LiveRunnerConfig,
    ):
        self.strategy = strategy
        self.notifier = notifier
        self.fetch_candles = fetch_candles
        self.config = config
        self._last_seen_timestamp: Optional[int] = None

    def step(self) -> Optional[Signal]:
        """Ejecuta un ciclo: obtiene velas, evalúa la última vela cerrada y
        notifica si corresponde. Devuelve la señal emitida, o None."""
        series = self.fetch_candles()
        if len(series) < self.strategy.warmup() + 1:
            return None

        i = len(series) - 1  # última vela ya cerrada
        last_candle = series[i]

        if self._last_seen_timestamp == last_candle.timestamp:
            return None  # esta vela ya fue procesada, evita señales duplicadas
        self._last_seen_timestamp = last_candle.timestamp

        if self.config.allowed_hours is not None:
            hour = (last_candle.timestamp // 3600) % 24
            if hour not in self.config.allowed_hours:
                return None

        signal = self.strategy.evaluate(series, i)
        if signal is None:
            return None

        signal.entry_price = last_candle.close
        message = format_signal_message(signal, self.config.symbol, self.config.interval)
        self.notifier.send(message)
        return signal

    def run_forever(self, max_iterations: Optional[int] = None) -> None:
        """Bucle de sondeo. `max_iterations` acota la ejecución (tests/uso
        controlado); `None` corre indefinidamente hasta interrupción."""
        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            try:
                self.step()
            except Exception as e:  # el runner no debe morir por un fallo puntual
                print(f"[error] fallo en el ciclo de señal: {e}")
            iterations += 1
            if max_iterations is None or iterations < max_iterations:
                time.sleep(self.config.poll_seconds)
