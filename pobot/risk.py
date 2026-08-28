"""Gestión de capital: tamaño de stake y límites diarios.

La martingala está soportada porque algunos usuarios la piden explícitamente,
pero viene desactivada por defecto y con advertencia: multiplicar el stake
tras una pérdida no cambia el EV esperado de la estrategia (que depende solo
del winrate y el payout), solo cambia la forma de la distribución de
resultados hacia "muchas ganancias pequeñas, ruina ocasional grande". Si la
estrategia no tiene ventaja demostrada, la martingala acelera la ruina.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pobot.config import RiskConfig
from pobot.edge import kelly_fraction


@dataclass
class RiskState:
    """Estado mutable de una sesión de trading (o de un backtest con gestión
    de capital), usado para calcular el stake de la siguiente operación."""

    balance: float
    starting_balance: float
    trades_today: int = 0
    pnl_today: float = 0.0
    consecutive_losses: int = 0

    def register_trade(self, pnl: float, was_loss: bool) -> None:
        self.balance += pnl
        self.pnl_today += pnl
        self.trades_today += 1
        self.consecutive_losses = self.consecutive_losses + 1 if was_loss else 0

    def reset_day(self) -> None:
        self.trades_today = 0
        self.pnl_today = 0.0


class RiskManager:
    def __init__(self, config: RiskConfig, payout: float, estimated_winrate: Optional[float] = None):
        self.config = config
        self.payout = payout
        self.estimated_winrate = estimated_winrate
        if config.martingale_enabled:
            import warnings

            warnings.warn(
                "Martingala activada: no mejora el EV esperado de la estrategia, "
                "solo cambia la forma del riesgo hacia pérdidas grandes poco frecuentes. "
                "Úsala solo si entiendes ese trade-off.",
                stacklevel=2,
            )

    def can_trade(self, state: RiskState) -> tuple[bool, str]:
        if self.config.max_trades_per_day is not None and state.trades_today >= self.config.max_trades_per_day:
            return False, "límite de operaciones diarias alcanzado"
        if self.config.daily_loss_limit is not None and state.pnl_today <= -abs(self.config.daily_loss_limit):
            return False, "límite de pérdida diaria alcanzado"
        if self.config.daily_profit_target is not None and state.pnl_today >= abs(self.config.daily_profit_target):
            return False, "objetivo de ganancia diaria alcanzado"
        return True, ""

    def next_stake(self, state: RiskState) -> float:
        base = self._base_stake(state)
        if not self.config.martingale_enabled or state.consecutive_losses == 0:
            return base
        steps = min(state.consecutive_losses, self.config.martingale_max_steps)
        return base * (self.config.martingale_multiplier**steps)

    def _base_stake(self, state: RiskState) -> float:
        mode = self.config.stake_mode
        if mode == "fixed":
            return self.config.fixed_stake
        if mode == "fraction":
            return state.balance * self.config.fraction
        if mode == "kelly":
            if self.estimated_winrate is None:
                raise ValueError("stake_mode='kelly' requiere estimated_winrate (del backtest/walk-forward)")
            f = kelly_fraction(self.estimated_winrate, self.payout, cap=self.config.kelly_cap)
            return state.balance * f
        raise ValueError(f"stake_mode desconocido: {mode!r}")
