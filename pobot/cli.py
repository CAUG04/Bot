"""CLI de pobot: `python3 -m pobot <comando> ...`

Comandos: download, backtest, walkforward, optimize, train, signal, live, report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pobot.backtest.engine import BacktestConfig, run_backtest
from pobot.backtest.metrics import compute_report
from pobot.backtest.optimizer import grid_search
from pobot.backtest.walkforward import run_walkforward
from pobot.config import BotConfig
from pobot.data.candles import CandleSeries
from pobot.data.candles import INTERVAL_SECONDS
from pobot.edge import breakeven_winrate
from pobot.features import FeatureBuilder
from pobot.labeling import make_labels
from pobot.live.notifier import MultiNotifier, format_signal_message
from pobot.live.runner import LiveRunner, LiveRunnerConfig
from pobot.ml.calibration import Calibrator
from pobot.ml.dataset import build_dataset, purged_split
from pobot.ml.logreg import LogisticRegression
from pobot.report import format_full_report
from pobot.strategies.base import available_strategies, get_strategy
from pobot.strategies.ml_strategy import MLStrategy


def _load_series(path: str) -> CandleSeries:
    return CandleSeries.from_csv(path)


def cmd_download(args: argparse.Namespace) -> int:
    from pobot.data.binance import download_klines, download_recent

    print(f"Descargando {args.symbol} {args.interval} desde {args.base_url} ...")
    if args.n_candles:
        series = download_recent(args.symbol, args.interval, args.n_candles, base_url=args.base_url)
    else:
        if args.start_ms is None:
            print("Error: especifica --n-candles o --start-ms", file=sys.stderr)
            return 2
        series = download_klines(args.symbol, args.interval, args.start_ms, args.end_ms, base_url=args.base_url)

    out_path = Path(args.out)
    series.to_csv(out_path)
    gaps = series.check_gaps(INTERVAL_SECONDS.get(args.interval))
    print(f"Guardadas {len(series)} velas en {out_path}")
    if gaps.has_gaps:
        print(f"Aviso: se detectaron {len(gaps.gaps)} huecos en la serie descargada.")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    series = _load_series(args.csv)
    params = json.loads(args.params) if args.params else {}
    strategy = get_strategy(args.strategy, **params)
    config = BacktestConfig(
        payout=args.payout,
        stake=args.stake,
        cooldown_bars=args.cooldown,
        max_trades_per_day=args.max_trades_per_day,
        tie_policy=args.tie_policy,
    )
    result = run_backtest(series, strategy, config)
    full = compute_report(result.trades, args.payout)
    print(format_full_report(full, title=f"Backtest: {args.strategy} sobre {args.csv}"))
    if args.save_trades:
        _save_trades_csv(result.trades, args.save_trades)
        print(f"\nTrades guardados en {args.save_trades}")
    return 0


def cmd_walkforward(args: argparse.Namespace) -> int:
    series = _load_series(args.csv)
    params = json.loads(args.params) if args.params else {}

    def factory(train_series):
        return get_strategy(args.strategy, **params)

    config = BacktestConfig(payout=args.payout, stake=args.stake, cooldown_bars=args.cooldown, tie_policy=args.tie_policy)
    full, trades, windows = run_walkforward(series, factory, config, args.train_size, args.test_size, args.step)
    print(format_full_report(full, title=f"Walk-forward: {args.strategy} sobre {args.csv} ({len(windows)} ventanas)"))
    if args.save_trades:
        _save_trades_csv(trades, args.save_trades)
        print(f"\nTrades guardados en {args.save_trades}")
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    series = _load_series(args.csv)
    grid = json.loads(Path(args.grid).read_text(encoding="utf-8"))

    def strategy_builder(params):
        def factory(train_series):
            return get_strategy(args.strategy, **params)

        return factory

    config = BacktestConfig(payout=args.payout, stake=args.stake, cooldown_bars=args.cooldown)
    results = grid_search(series, grid, strategy_builder, config, args.train_size, args.test_size, args.step)

    print(f"Optimización de '{args.strategy}': {len(results)} combinaciones probadas (out-of-sample)\n")
    for r in results[: args.top]:
        edge = "VENTAJA (corregida por nº de pruebas)" if r.corrected_has_edge else "sin ventaja"
        print(
            f"  params={r.params}  n={r.report.overall.n_trades}  winrate={r.report.overall.winrate:.2%}  "
            f"z={r.z_used}  {edge}"
        )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    series = _load_series(args.csv)
    horizon = args.horizon
    rows = FeatureBuilder().build(series)
    labels = make_labels(series, horizon=horizon, tie_policy=args.tie_policy)
    dataset = build_dataset(rows, labels)
    if len(dataset) < 50:
        print(f"Aviso: dataset pequeño ({len(dataset)} filas), el modelo puede no ser fiable.")

    split_index = int(len(series) * args.train_fraction)
    train, test = purged_split(dataset, split_index, horizon)
    print(f"Train: {len(train)} filas | Test (OOS): {len(test)} filas | horizon={horizon}")

    model = LogisticRegression(lr=args.lr, epochs=args.epochs, seed=args.seed)
    val_split = int(len(train.X) * 0.85)
    model.fit(train.X[:val_split], train.y[:val_split], train.X[val_split:], train.y[val_split:])

    probs_train = model.predict_proba(train.X)
    calibrator = Calibrator.fit(probs_train, train.y, n_bins=args.calibration_bins)

    if test.X:
        from pobot.edge import has_demonstrated_edge

        probs_test = calibrator.calibrate_many(model.predict_proba(test.X))
        preds = [1 if p >= 0.5 else 0 for p in probs_test]
        wins = sum(1 for p, y in zip(preds, test.y) if p == y)
        n = len(test.y)
        be = breakeven_winrate(args.payout)
        print(f"Winrate OOS del modelo: {wins/n:.2%} (breakeven={be:.2%})")
        print(f"¿Ventaja demostrada (Wilson)?: {'SÍ' if has_demonstrated_edge(wins, n, args.payout) else 'NO'}")

    out = {"model": model.to_dict(), "calibrator": calibrator.to_dict(), "horizon": horizon, "payout": args.payout}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nModelo guardado en {args.out}")
    return 0


def _load_ml_strategy(model_path: str, expiry_bars: int | None = None, min_confidence: float | None = None) -> MLStrategy:
    data = json.loads(Path(model_path).read_text(encoding="utf-8"))
    model = LogisticRegression.from_dict(data["model"])
    calibrator = Calibrator.from_dict(data["calibrator"])
    return MLStrategy(
        model=model,
        calibrator=calibrator,
        payout=data.get("payout", 0.85),
        expiry_bars=expiry_bars or data.get("horizon", 3),
        min_confidence=min_confidence,
    )


def cmd_signal(args: argparse.Namespace) -> int:
    series = _load_series(args.csv)
    params = json.loads(args.params) if args.params else {}
    if args.strategy == "ml":
        strategy = _load_ml_strategy(args.model)
    else:
        strategy = get_strategy(args.strategy, **params)

    i = len(series) - 1
    if i < strategy.warmup():
        print("No hay suficiente historial para evaluar la última vela.", file=sys.stderr)
        return 1

    signal = strategy.evaluate(series, i)
    if signal is None:
        print("Sin señal en la última vela cerrada.")
        return 0

    signal.entry_price = series[i].close
    message = format_signal_message(signal, args.symbol, args.interval)
    cfg = BotConfig.load()
    notifier = MultiNotifier(cfg.telegram if args.telegram else None)
    notifier.send(message)
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from pobot.data.binance import download_recent

    params = json.loads(args.params) if args.params else {}
    if args.strategy == "ml":
        strategy = _load_ml_strategy(args.model)
    else:
        strategy = get_strategy(args.strategy, **params)

    cfg = BotConfig.load()
    notifier = MultiNotifier(cfg.telegram if args.telegram else None)

    def fetch():
        return download_recent(args.symbol, args.interval, args.lookback, base_url=args.base_url)

    runner_config = LiveRunnerConfig(
        symbol=args.symbol, interval=args.interval, payout=args.payout, poll_seconds=args.poll_seconds
    )
    runner = LiveRunner(strategy, notifier, fetch, runner_config)
    print(f"Runner en vivo para {args.symbol} {args.interval}. Ctrl+C para detener.")
    try:
        runner.run_forever()
    except KeyboardInterrupt:
        print("\nDetenido por el usuario.")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from pobot.types import Trade

    trades = _load_trades_csv(args.trades_csv)
    full = compute_report(trades, args.payout)
    print(format_full_report(full, title=f"Reporte: {args.trades_csv}"))
    return 0


def cmd_list_strategies(args: argparse.Namespace) -> int:
    for name in available_strategies():
        print(name)
    return 0


def _save_trades_csv(trades, path: str) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["entry_timestamp", "exit_timestamp", "direction", "entry_price", "exit_price",
             "expiry_bars", "stake", "payout", "strategy", "confidence", "result", "pnl"]
        )
        for t in trades:
            writer.writerow(
                [t.entry_timestamp, t.exit_timestamp, t.direction.value, t.entry_price, t.exit_price,
                 t.expiry_bars, t.stake, t.payout, t.strategy, t.confidence, t.result.value, t.pnl]
            )


def _load_trades_csv(path: str):
    import csv

    from pobot.types import Direction, Trade

    trades = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            trades.append(
                Trade(
                    entry_index=0,
                    entry_timestamp=int(row["entry_timestamp"]),
                    exit_index=0,
                    exit_timestamp=int(row["exit_timestamp"]),
                    direction=Direction(row["direction"]),
                    entry_price=float(row["entry_price"]),
                    exit_price=float(row["exit_price"]),
                    expiry_bars=int(row["expiry_bars"]),
                    stake=float(row["stake"]),
                    payout=float(row["payout"]),
                    strategy=row["strategy"],
                    confidence=float(row["confidence"]),
                )
            )
    return trades


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pobot", description="Señales de trading para opciones binarias")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="Descarga velas de Binance a un CSV")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1m")
    p.add_argument("--n-candles", type=int, dest="n_candles", default=None)
    p.add_argument("--start-ms", type=int, dest="start_ms", default=None)
    p.add_argument("--end-ms", type=int, dest="end_ms", default=None)
    p.add_argument("--base-url", dest="base_url", default="https://api.binance.com")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("backtest", help="Corre un backtest sobre un CSV de velas")
    p.add_argument("--csv", required=True)
    p.add_argument("--strategy", default="confluence")
    p.add_argument("--params", default=None, help="JSON con parámetros del constructor de la estrategia")
    p.add_argument("--payout", type=float, default=0.85)
    p.add_argument("--stake", type=float, default=1.0)
    p.add_argument("--cooldown", type=int, default=1)
    p.add_argument("--max-trades-per-day", type=int, dest="max_trades_per_day", default=None)
    p.add_argument("--tie-policy", dest="tie_policy", default="refund", choices=["refund", "loss"])
    p.add_argument("--save-trades", dest="save_trades", default=None)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("walkforward", help="Validación out-of-sample rodante")
    p.add_argument("--csv", required=True)
    p.add_argument("--strategy", default="confluence")
    p.add_argument("--params", default=None)
    p.add_argument("--payout", type=float, default=0.85)
    p.add_argument("--stake", type=float, default=1.0)
    p.add_argument("--cooldown", type=int, default=1)
    p.add_argument("--tie-policy", dest="tie_policy", default="refund", choices=["refund", "loss"])
    p.add_argument("--train-size", type=int, dest="train_size", required=True)
    p.add_argument("--test-size", type=int, dest="test_size", required=True)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--save-trades", dest="save_trades", default=None)
    p.set_defaults(func=cmd_walkforward)

    p = sub.add_parser("optimize", help="Búsqueda de parámetros puntuada out-of-sample")
    p.add_argument("--csv", required=True)
    p.add_argument("--strategy", default="confluence")
    p.add_argument("--grid", required=True, help="Ruta a un JSON {param: [valores]}")
    p.add_argument("--payout", type=float, default=0.85)
    p.add_argument("--stake", type=float, default=1.0)
    p.add_argument("--cooldown", type=int, default=1)
    p.add_argument("--train-size", type=int, dest="train_size", required=True)
    p.add_argument("--test-size", type=int, dest="test_size", required=True)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("train", help="Entrena el modelo ML (regresión logística) y lo guarda")
    p.add_argument("--csv", required=True)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--payout", type=float, default=0.85)
    p.add_argument("--tie-policy", dest="tie_policy", default="refund", choices=["refund", "loss"])
    p.add_argument("--train-fraction", type=float, dest="train_fraction", default=0.8)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calibration-bins", type=int, dest="calibration_bins", default=10)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("signal", help="Evalúa la última vela cerrada de un CSV y emite una señal")
    p.add_argument("--csv", required=True)
    p.add_argument("--strategy", default="confluence")
    p.add_argument("--params", default=None)
    p.add_argument("--model", default=None, help="Ruta al modelo entrenado (requerido si --strategy=ml)")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1m")
    p.add_argument("--telegram", action="store_true")
    p.set_defaults(func=cmd_signal)

    p = sub.add_parser("live", help="Sondea Binance en vivo y emite señales")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1m")
    p.add_argument("--strategy", default="confluence")
    p.add_argument("--params", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--payout", type=float, default=0.85)
    p.add_argument("--lookback", type=int, default=200)
    p.add_argument("--poll-seconds", type=float, dest="poll_seconds", default=5.0)
    p.add_argument("--base-url", dest="base_url", default="https://api.binance.com")
    p.add_argument("--telegram", action="store_true")
    p.set_defaults(func=cmd_live)

    p = sub.add_parser("report", help="Regenera el reporte de texto a partir de un CSV de trades")
    p.add_argument("--trades-csv", required=True, dest="trades_csv")
    p.add_argument("--payout", type=float, default=0.85)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("list-strategies", help="Lista las estrategias registradas")
    p.set_defaults(func=cmd_list_strategies)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
