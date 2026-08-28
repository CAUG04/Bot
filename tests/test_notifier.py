"""Prueba el notifier de Telegram contra un servidor HTTP local.

No se envía nada a la API real de Telegram: se valida el formateo del
mensaje y el payload HTTP contra un stub local, y que un fallo de red no
tumbe el `MultiNotifier` (Telegram es un canal opcional).
"""

import contextlib
import io
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from pobot.config import TelegramConfig
from pobot.live.notifier import ConsoleNotifier, MultiNotifier, TelegramNotifier, format_signal_message
from pobot.types import Direction, Signal


def _signal():
    return Signal(
        index=10,
        timestamp=1_700_000_000,
        direction=Direction.CALL,
        confidence=0.63,
        expiry_bars=3,
        strategy="confluence",
        reasons=["RSI en sobreventa", "EMA9 > EMA21"],
        entry_price=64123.5,
    )


class TestFormatSignalMessage(unittest.TestCase):
    def test_includes_key_fields(self):
        msg = format_signal_message(_signal(), symbol="BTCUSDT", interval="1m")
        self.assertIn("BTCUSDT", msg)
        self.assertIn("CALL", msg)
        self.assertIn("63.0%", msg)
        self.assertIn("3 vela", msg)
        self.assertIn("RSI en sobreventa", msg)


class TestConsoleNotifier(unittest.TestCase):
    def test_prints_message(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ConsoleNotifier().send("hola mundo")
        self.assertIn("hola mundo", buf.getvalue())


class _CapturingTelegramHandler(BaseHTTPRequestHandler):
    received: list[dict] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self.received.append(json.loads(body))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))

    def log_message(self, fmt, *args):
        pass


class TestTelegramNotifier(unittest.TestCase):
    def _start_server(self):
        _CapturingTelegramHandler.received = []
        server = HTTPServer(("127.0.0.1", 0), _CapturingTelegramHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        return server, f"http://127.0.0.1:{port}"

    def test_sends_chat_id_and_text(self):
        server, base_url = self._start_server()
        try:
            config = TelegramConfig(bot_token="TESTTOKEN", chat_id="12345")
            notifier = TelegramNotifier(config, api_base=base_url)
            notifier.send("mensaje de prueba")
        finally:
            server.shutdown()
        self.assertEqual(len(_CapturingTelegramHandler.received), 1)
        payload = _CapturingTelegramHandler.received[0]
        self.assertEqual(payload["chat_id"], "12345")
        self.assertEqual(payload["text"], "mensaje de prueba")

    def test_raises_without_config(self):
        with self.assertRaises(ValueError):
            TelegramNotifier(TelegramConfig())


class TestMultiNotifier(unittest.TestCase):
    def test_console_only_when_telegram_not_configured(self):
        notifier = MultiNotifier(TelegramConfig())
        self.assertIsNone(notifier.telegram)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            notifier.send("solo consola")
        self.assertIn("solo consola", buf.getvalue())

    def test_sends_to_both_channels_when_configured(self):
        server = HTTPServer(("127.0.0.1", 0), _CapturingTelegramHandler)
        _CapturingTelegramHandler.received = []
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        try:
            config = TelegramConfig(bot_token="T", chat_id="1")
            notifier = MultiNotifier(config)
            notifier.telegram.api_base = f"http://127.0.0.1:{port}"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                notifier.send("doble canal")
        finally:
            server.shutdown()
        self.assertIn("doble canal", buf.getvalue())
        self.assertEqual(len(_CapturingTelegramHandler.received), 1)

    def test_telegram_failure_does_not_raise(self):
        config = TelegramConfig(bot_token="T", chat_id="1")
        notifier = MultiNotifier(config)
        notifier.telegram.api_base = "http://127.0.0.1:1"  # puerto que rechaza conexión
        notifier.telegram.timeout = 1.0
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            notifier.send("no debe lanzar excepción")  # no debe propagar el error de red
        self.assertIn("no debe lanzar excepción", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
