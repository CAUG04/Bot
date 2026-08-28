"""Notificación de señales: consola siempre, Telegram si está configurado.

Usa solo `urllib` (stdlib): sin dependencias externas. Si no hay token/chat_id
configurados, el bot sigue funcionando en modo consola sin fallar — Telegram
es un canal opcional, no un requisito para operar el bot.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from pobot.config import TelegramConfig
from pobot.types import Signal

TELEGRAM_API_BASE = "https://api.telegram.org"


def format_signal_message(signal: Signal, symbol: str, interval: str) -> str:
    direction_label = "🔼 CALL (sube)" if signal.direction.value == "CALL" else "🔽 PUT (baja)"
    lines = [
        f"Señal: {symbol} ({interval})",
        f"Dirección: {direction_label}",
        f"Confianza estimada: {signal.confidence:.1%}",
        f"Expiración: {signal.expiry_bars} vela(s)",
        f"Estrategia: {signal.strategy}",
    ]
    if signal.entry_price is not None:
        lines.append(f"Precio de referencia: {signal.entry_price}")
    if signal.reasons:
        lines.append("Motivos: " + "; ".join(signal.reasons))
    return "\n".join(lines)


class ConsoleNotifier:
    def send(self, message: str) -> None:
        print(message)


class TelegramNotifier:
    def __init__(self, config: TelegramConfig, timeout: float = 10.0, api_base: str = TELEGRAM_API_BASE):
        if not config.enabled:
            raise ValueError("TelegramConfig incompleto: faltan bot_token o chat_id")
        self.config = config
        self.timeout = timeout
        self.api_base = api_base

    def send(self, message: str) -> None:
        url = f"{self.api_base}/bot{self.config.bot_token}/sendMessage"
        payload = json.dumps({"chat_id": self.config.chat_id, "text": message}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp.read()
        except urllib.error.URLError as e:
            raise RuntimeError(f"No se pudo enviar el mensaje a Telegram: {e}") from e


class MultiNotifier:
    """Envía siempre por consola y, si Telegram está configurado, también por ahí.

    Un fallo de red al enviar a Telegram no debe tumbar el bot: se registra
    el error y se sigue operando en modo consola.
    """

    def __init__(self, telegram_config: TelegramConfig | None = None):
        self.console = ConsoleNotifier()
        self.telegram: TelegramNotifier | None = None
        if telegram_config is not None and telegram_config.enabled:
            self.telegram = TelegramNotifier(telegram_config)

    def send(self, message: str) -> None:
        self.console.send(message)
        if self.telegram is not None:
            try:
                self.telegram.send(message)
            except RuntimeError as e:
                print(f"[aviso] no se pudo notificar por Telegram: {e}")
