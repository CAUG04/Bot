"""Prueba el descargador de Binance contra un servidor HTTP local.

El proxy de red de esta sesión de desarrollo bloquea api.binance.com, así que
no se puede probar contra la API real aquí. Este test valida la lógica de
paginación, deduplicado y parseo con un servidor stdlib que responde con el
mismo formato que Binance. La descarga real contra `api.binance.com` queda
documentada en el README para ejecutarse en un entorno con salida a internet.
"""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from pobot.data.binance import download_klines, BinanceDownloadError


def _make_kline_row(open_time_ms: int, o: float, h: float, l: float, c: float, v: float, step_ms: int):
    close_time_ms = open_time_ms + step_ms - 1
    return [
        open_time_ms, f"{o}", f"{h}", f"{l}", f"{c}", f"{v}",
        close_time_ms, "0", 0, "0", "0", "0",
    ]


class _FakeBinanceHandler(BaseHTTPRequestHandler):
    # Generado en la clase de test vía atributos de clase inyectados dinámicamente
    pages: list[list[list]] = []
    call_log: list[dict] = []

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        self.call_log.append({k: v[0] for k, v in qs.items()})
        idx = len(self.call_log) - 1
        page = self.pages[idx] if idx < len(self.pages) else []
        body = json.dumps(page).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # silenciar logs del servidor de test


class TestBinanceDownloader(unittest.TestCase):
    def _start_server(self, pages):
        _FakeBinanceHandler.pages = pages
        _FakeBinanceHandler.call_log = []
        server = HTTPServer(("127.0.0.1", 0), _FakeBinanceHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        return server, thread, f"http://127.0.0.1:{port}"

    def test_single_page_download(self):
        step_ms = 60_000
        page = [
            _make_kline_row(1_700_000_000_000 + i * step_ms, 100 + i, 101 + i, 99 + i, 100.5 + i, 10, step_ms)
            for i in range(3)
        ]
        server, thread, base_url = self._start_server([page])
        try:
            series = download_klines(
                "BTCUSDT", "1m",
                start_ms=1_700_000_000_000,
                end_ms=1_700_000_000_000 + 3 * step_ms,
                base_url=base_url,
                sleep_between_pages=0,
            )
        finally:
            server.shutdown()
        self.assertEqual(len(series), 3)
        self.assertEqual(series[0].open, 100.0)
        self.assertEqual(series[-1].close, 102.5)

    def test_pagination_across_two_pages(self):
        step_ms = 60_000
        page1 = [
            _make_kline_row(1_700_000_000_000 + i * step_ms, 100 + i, 101 + i, 99 + i, 100.5 + i, 10, step_ms)
            for i in range(1000)
        ]
        page2 = [
            _make_kline_row(1_700_000_000_000 + (1000 + i) * step_ms, 200 + i, 201 + i, 199 + i, 200.5 + i, 10, step_ms)
            for i in range(5)
        ]
        server, thread, base_url = self._start_server([page1, page2])
        try:
            series = download_klines(
                "BTCUSDT", "1m",
                start_ms=1_700_000_000_000,
                end_ms=None,
                base_url=base_url,
                sleep_between_pages=0,
            )
        finally:
            server.shutdown()
        self.assertEqual(len(series), 1005)
        self.assertEqual(len(_FakeBinanceHandler.call_log), 2)

    def test_http_error_raises_binance_download_error(self):
        class _ErrHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(429)
                self.end_headers()

            def log_message(self, fmt, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), _ErrHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        try:
            with self.assertRaises(BinanceDownloadError):
                download_klines(
                    "BTCUSDT", "1m",
                    start_ms=0, end_ms=60_000,
                    base_url=f"http://127.0.0.1:{port}",
                    max_retries=1,
                    sleep_between_pages=0,
                )
        finally:
            server.shutdown()

    def test_unsupported_interval_raises(self):
        with self.assertRaises(ValueError):
            download_klines("BTCUSDT", "2m", start_ms=0)


if __name__ == "__main__":
    unittest.main()
