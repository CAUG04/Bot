"""Prueba la CLI extremo a extremo: generar CSV sintético -> backtest ->
walkforward -> train -> signal -> report. Todo offline, sin red.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from pobot.cli import main
from pobot.data.candles import CandleSeries
from pobot.data.synthetic import random_walk_candles


class TestCLIEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.csv_path = str(Path(self.tmpdir.name) / "candles.csv")
        candles = random_walk_candles(500, seed=17, drift=0.0003)
        CandleSeries(candles).to_csv(self.csv_path)

    def _run(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def test_list_strategies(self):
        code, out = self._run(["list-strategies"])
        self.assertEqual(code, 0)
        self.assertIn("confluence", out)
        self.assertIn("rsi_reversal", out)

    def test_backtest_runs_and_prints_report(self):
        code, out = self._run(
            ["backtest", "--csv", self.csv_path, "--strategy", "rsi_reversal", "--payout", "0.85"]
        )
        self.assertEqual(code, 0)
        self.assertIn("Winrate observado", out)
        self.assertIn("ventaja", out.lower())

    def test_backtest_with_params_and_save_trades(self):
        trades_path = str(Path(self.tmpdir.name) / "trades.csv")
        params = json.dumps({"period": 10, "oversold": 25, "overbought": 75})
        code, out = self._run(
            [
                "backtest", "--csv", self.csv_path, "--strategy", "rsi_reversal",
                "--params", params, "--save-trades", trades_path,
            ]
        )
        self.assertEqual(code, 0)
        self.assertTrue(Path(trades_path).exists())

    def test_walkforward_runs(self):
        code, out = self._run(
            [
                "walkforward", "--csv", self.csv_path, "--strategy", "ema_trend",
                "--train-size", "100", "--test-size", "50",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("Walk-forward", out)

    def test_optimize_runs_with_grid(self):
        grid_path = str(Path(self.tmpdir.name) / "grid.json")
        Path(grid_path).write_text(json.dumps({"period": [10, 14], "oversold": [25, 30]}))
        code, out = self._run(
            [
                "optimize", "--csv", self.csv_path, "--strategy", "rsi_reversal",
                "--grid", grid_path, "--train-size", "100", "--test-size", "50", "--top", "5",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("combinaciones probadas", out)

    def test_train_produces_model_file(self):
        model_path = str(Path(self.tmpdir.name) / "model.json")
        code, out = self._run(
            ["train", "--csv", self.csv_path, "--horizon", "3", "--epochs", "20", "--out", model_path]
        )
        self.assertEqual(code, 0)
        self.assertTrue(Path(model_path).exists())
        data = json.loads(Path(model_path).read_text())
        self.assertIn("model", data)
        self.assertIn("calibrator", data)

    def test_signal_with_rule_strategy_prints_console(self):
        code, out = self._run(
            ["signal", "--csv", self.csv_path, "--strategy", "ema_trend", "--symbol", "BTCUSDT"]
        )
        self.assertEqual(code, 0)
        self.assertTrue("Sin señal" in out or "BTCUSDT" in out)

    def test_signal_with_ml_strategy_uses_trained_model(self):
        model_path = str(Path(self.tmpdir.name) / "model_signal.json")
        code, _ = self._run(
            ["train", "--csv", self.csv_path, "--horizon", "3", "--epochs", "20", "--out", model_path]
        )
        self.assertEqual(code, 0)
        code, out = self._run(
            ["signal", "--csv", self.csv_path, "--strategy", "ml", "--model", model_path, "--symbol", "BTCUSDT"]
        )
        self.assertEqual(code, 0)
        self.assertTrue("Sin señal" in out or "BTCUSDT" in out)

    def test_report_regenerates_from_trades_csv(self):
        trades_path = str(Path(self.tmpdir.name) / "trades_for_report.csv")
        self._run(["backtest", "--csv", self.csv_path, "--strategy", "confluence", "--save-trades", trades_path])
        code, out = self._run(["report", "--trades-csv", trades_path, "--payout", "0.85"])
        self.assertEqual(code, 0)
        self.assertIn("Winrate observado", out)

    def test_download_without_source_returns_error(self):
        out_path = str(Path(self.tmpdir.name) / "should_not_exist.csv")
        code, out = self._run(["download", "--symbol", "BTCUSDT", "--interval", "1m", "--out", out_path])
        self.assertEqual(code, 2)
        self.assertFalse(Path(out_path).exists())


if __name__ == "__main__":
    unittest.main()
