import json
import os
import tempfile
import unittest
from pathlib import Path

from pobot.config import BotConfig, load_dotenv, TelegramConfig


class TestDotenv(unittest.TestCase):
    def setUp(self):
        self._saved = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved)

    def test_load_dotenv_parses_and_sets_env(self):
        with tempfile.TemporaryDirectory() as d:
            env_path = Path(d) / ".env"
            env_path.write_text(
                "# comentario\n"
                "TELEGRAM_BOT_TOKEN=abc123\n"
                'TELEGRAM_CHAT_ID="999"\n'
                "\n"
                "NOT_A_LINE_WITHOUT_EQUALS\n"
            )
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
            os.environ.pop("TELEGRAM_CHAT_ID", None)
            loaded = load_dotenv(env_path)
            self.assertEqual(loaded["TELEGRAM_BOT_TOKEN"], "abc123")
            self.assertEqual(os.environ["TELEGRAM_CHAT_ID"], "999")

    def test_load_dotenv_missing_file_returns_empty(self):
        loaded = load_dotenv("/tmp/does-not-exist-pobot.env")
        self.assertEqual(loaded, {})

    def test_env_does_not_override_existing(self):
        with tempfile.TemporaryDirectory() as d:
            env_path = Path(d) / ".env"
            env_path.write_text("TELEGRAM_BOT_TOKEN=fromfile\n")
            os.environ["TELEGRAM_BOT_TOKEN"] = "already-set"
            load_dotenv(env_path)
            self.assertEqual(os.environ["TELEGRAM_BOT_TOKEN"], "already-set")


class TestBotConfig(unittest.TestCase):
    def test_default_config_has_sane_values(self):
        cfg = BotConfig()
        self.assertEqual(cfg.market.symbol, "BTCUSDT")
        self.assertIn(1, cfg.strategy.expiry_candidates)
        self.assertFalse(cfg.telegram.enabled)

    def test_telegram_enabled_requires_both_fields(self):
        self.assertFalse(TelegramConfig(bot_token="x").enabled)
        self.assertFalse(TelegramConfig(chat_id="y").enabled)
        self.assertTrue(TelegramConfig(bot_token="x", chat_id="y").enabled)

    def test_save_never_persists_telegram_secrets(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = BotConfig(telegram=TelegramConfig(bot_token="secret", chat_id="123"))
            out = Path(d) / "config.json"
            cfg.save(out)
            data = json.loads(out.read_text())
            self.assertNotIn("telegram", data)

    def test_load_json_overrides_market(self):
        with tempfile.TemporaryDirectory() as d:
            cfg_path = Path(d) / "config.json"
            cfg_path.write_text(json.dumps({"market": {"symbol": "ETHUSDT", "interval": "5m", "payout": 0.9}}))
            cfg = BotConfig.load(cfg_path, env_path=Path(d) / "nope.env")
            self.assertEqual(cfg.market.symbol, "ETHUSDT")
            self.assertEqual(cfg.market.interval, "5m")


if __name__ == "__main__":
    unittest.main()
