"""Configuración del bot: dataclasses + carga desde JSON y `.env`.

No usa dependencias externas: el parser de `.env` es propio (KEY=VALUE por
línea, `#` para comentarios, comillas opcionales).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Carga variables de un archivo .env al entorno del proceso (no sobreescribe
    variables ya presentes en `os.environ`). Devuelve lo que cargó."""
    p = Path(path)
    loaded: dict[str, str] = {}
    if not p.exists():
        return loaded
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded


@dataclass
class TelegramConfig:
    bot_token: str = ""
    chat_id: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    @classmethod
    def from_env(cls) -> "TelegramConfig":
        return cls(
            bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
        )


@dataclass
class MarketConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    payout: float = 0.85  # payout típico de Pocket Option en cripto/OTC; ajustar por activo


@dataclass
class StrategyConfig:
    name: str = "confluence"
    expiry_candidates: list[int] = field(default_factory=lambda: [1, 2, 3, 5])
    min_confidence: float = 0.55
    params: dict = field(default_factory=dict)


@dataclass
class RiskConfig:
    stake_mode: str = "fixed"  # fixed | fraction | kelly
    fixed_stake: float = 1.0
    fraction: float = 0.01
    kelly_cap: float = 0.25
    max_trades_per_day: int = 20
    daily_loss_limit: Optional[float] = None
    daily_profit_target: Optional[float] = None
    martingale_enabled: bool = False
    martingale_multiplier: float = 2.2
    martingale_max_steps: int = 3


@dataclass
class BacktestConfig:
    cooldown_bars: int = 1
    allowed_hours: Optional[list[int]] = None  # None = todas las horas UTC
    tie_policy: str = "refund"  # refund | loss


@dataclass
class BotConfig:
    market: MarketConfig = field(default_factory=MarketConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    data_dir: str = "data"
    models_dir: str = "models"

    @classmethod
    def load(cls, path: Optional[str | Path] = None, env_path: str | Path = ".env") -> "BotConfig":
        load_dotenv(env_path)
        cfg = cls(telegram=TelegramConfig.from_env())
        if path is None:
            return cfg
        p = Path(path)
        if not p.exists():
            return cfg
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            market=MarketConfig(**data.get("market", {})),
            strategy=StrategyConfig(**data.get("strategy", {})),
            risk=RiskConfig(**data.get("risk", {})),
            backtest=BacktestConfig(**data.get("backtest", {})),
            telegram=cfg.telegram,
            data_dir=data.get("data_dir", cfg.data_dir),
            models_dir=data.get("models_dir", cfg.models_dir),
        )

    def save(self, path: str | Path) -> None:
        data = asdict(self)
        data.pop("telegram", None)  # nunca persistir secretos en el JSON de config
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
