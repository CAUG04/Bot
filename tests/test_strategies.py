import unittest

from pobot.data.candles import CandleSeries
from pobot.data.synthetic import random_walk_candles, trending_candles
from pobot.strategies.base import Vote, VotingStrategy, get_strategy, available_strategies
from pobot.strategies.bollinger_bounce import BollingerBounceStrategy
from pobot.strategies.confluence import ConfluenceStrategy
from pobot.strategies.ema_trend import EMATrendStrategy
from pobot.strategies.macd_momentum import MACDMomentumStrategy
from pobot.strategies.pattern_sr import PatternSRStrategy
from pobot.strategies.rsi_reversal import RSIReversalStrategy
from pobot.types import Candle, Direction


def _c(ts, o, h, l, c, v=100.0):
    return Candle(timestamp=ts, open=o, high=h, low=l, close=c, volume=v)


class TestRegistry(unittest.TestCase):
    def test_all_strategies_are_registered(self):
        names = available_strategies()
        for expected in ["rsi_reversal", "ema_trend", "bollinger_bounce", "macd_momentum", "pattern_sr", "confluence"]:
            self.assertIn(expected, names)

    def test_get_strategy_instantiates(self):
        strat = get_strategy("rsi_reversal", period=10)
        self.assertEqual(strat.period, 10)

    def test_get_strategy_unknown_raises(self):
        with self.assertRaises(KeyError):
            get_strategy("does_not_exist")


class TestRSIReversal(unittest.TestCase):
    def test_votes_call_after_sharp_decline(self):
        # caída sostenida -> RSI bajo -> voto CALL (sobreventa)
        closes = list(range(100, 60, -1))
        candles = [_c(i * 60, c, c + 1, c - 1, c) for i, c in enumerate(closes)]
        series = CandleSeries(candles)
        strat = RSIReversalStrategy(period=14, min_confidence=0.0)
        vote = strat.vote(series, len(series) - 1)
        self.assertIsNotNone(vote)
        self.assertEqual(vote.direction, Direction.CALL)

    def test_votes_put_after_sharp_rally(self):
        closes = list(range(60, 100))
        candles = [_c(i * 60, c, c + 1, c - 1, c) for i, c in enumerate(closes)]
        series = CandleSeries(candles)
        strat = RSIReversalStrategy(period=14, min_confidence=0.0)
        vote = strat.vote(series, len(series) - 1)
        self.assertIsNotNone(vote)
        self.assertEqual(vote.direction, Direction.PUT)

    def test_no_vote_before_warmup(self):
        candles = [_c(i * 60, 10, 11, 9, 10) for i in range(5)]
        series = CandleSeries(candles)
        strat = RSIReversalStrategy(period=14)
        self.assertIsNone(strat.vote(series, 2))


class TestEMATrend(unittest.TestCase):
    def test_votes_call_on_strong_uptrend(self):
        candles = trending_candles(100, trend_strength=0.004, volatility=0.0005, seed=1)
        series = CandleSeries(candles)
        strat = EMATrendStrategy(min_confidence=0.0)
        vote = strat.vote(series, len(series) - 1)
        self.assertIsNotNone(vote)
        self.assertEqual(vote.direction, Direction.CALL)

    def test_votes_put_on_strong_downtrend(self):
        candles = trending_candles(100, trend_strength=-0.004, volatility=0.0005, seed=2)
        series = CandleSeries(candles)
        strat = EMATrendStrategy(min_confidence=0.0)
        vote = strat.vote(series, len(series) - 1)
        self.assertIsNotNone(vote)
        self.assertEqual(vote.direction, Direction.PUT)


class TestBollingerBounce(unittest.TestCase):
    def test_votes_call_when_price_pierces_lower_band(self):
        closes = [100.0] * 25 + [80.0]  # caída brusca fuera de la banda inferior
        candles = [_c(i * 60, c, c + 0.5, c - 0.5, c) for i, c in enumerate(closes)]
        series = CandleSeries(candles)
        strat = BollingerBounceStrategy(period=20, min_confidence=0.0)
        vote = strat.vote(series, len(series) - 1)
        self.assertIsNotNone(vote)
        self.assertEqual(vote.direction, Direction.CALL)

    def test_votes_put_when_price_pierces_upper_band(self):
        closes = [100.0] * 25 + [120.0]
        candles = [_c(i * 60, c, c + 0.5, c - 0.5, c) for i, c in enumerate(closes)]
        series = CandleSeries(candles)
        strat = BollingerBounceStrategy(period=20, min_confidence=0.0)
        vote = strat.vote(series, len(series) - 1)
        self.assertIsNotNone(vote)
        self.assertEqual(vote.direction, Direction.PUT)

    def test_no_vote_when_price_inside_bands(self):
        candles = [_c(i * 60, 100, 100.5, 99.5, 100) for i in range(25)]
        series = CandleSeries(candles)
        strat = BollingerBounceStrategy(period=20, min_confidence=0.0)
        self.assertIsNone(strat.vote(series, len(series) - 1))


class TestMACDMomentum(unittest.TestCase):
    def test_votes_call_on_bullish_crossover(self):
        # Nota: una V perfectamente lineal (pendiente constante a cada lado)
        # es degenerada para MACD: EMA rápida y lenta convergen a un offset
        # fijo y el histograma queda ~constante, sin cruces reales. Se usa
        # una caída que se acelera seguida de una subida que se acelera
        # (curvatura real) para producir un cruce de histograma genuino, muy
        # por encima del warmup de la estrategia.
        n = 80
        down = [150 - i * 0.8 - (i**1.6) * 0.05 for i in range(n)]
        trough = down[-1]
        up = [trough + j * 0.8 + (j**1.6) * 0.05 for j in range(1, n + 1)]
        closes = [float(x) for x in down + up]
        candles = [_c(i * 60, c, c + 0.3, c - 0.3, c) for i, c in enumerate(closes)]
        series = CandleSeries(candles)
        strat = MACDMomentumStrategy(min_confidence=0.0)
        votes = [strat.vote(series, i) for i in range(strat.warmup(), len(series))]
        call_votes = [v for v in votes if v is not None and v.direction is Direction.CALL]
        self.assertTrue(len(call_votes) > 0)


class TestPatternSR(unittest.TestCase):
    def test_votes_call_on_pin_bar_at_support(self):
        # canal lateral que forma un mínimo estable, luego un pin bar alcista
        # justo en ese nivel de soporte
        base = [_c(i * 60, 100, 101, 99, 100) for i in range(20)]
        pin_bar = _c(20 * 60, 99.5, 99.6, 95.0, 99.4)  # mecha inferior larga, cierra cerca del soporte
        series = CandleSeries(base + [pin_bar])
        strat = PatternSRStrategy(donchian_period=20, proximity_pct=0.3, min_confidence=0.0)
        vote = strat.vote(series, len(series) - 1)
        self.assertIsNotNone(vote)
        self.assertEqual(vote.direction, Direction.CALL)


class _FixedVoter(VotingStrategy):
    name = "fixed_voter_test"

    def __init__(self, direction, strength):
        super().__init__(min_confidence=0.0)
        self._direction = direction
        self._strength = strength

    def vote(self, series, i):
        return Vote(self._direction, self._strength, ["fixed"])


class TestConfluence(unittest.TestCase):
    def test_no_signal_when_voters_disagree_evenly(self):
        voters = [(_FixedVoter(Direction.CALL, 0.6), 1.0), (_FixedVoter(Direction.PUT, 0.6), 1.0)]
        strat = ConfluenceStrategy(voters=voters, min_voters=1, min_net_strength=0.55)
        candles = [_c(0, 10, 10, 10, 10)]
        series = CandleSeries(candles)
        self.assertIsNone(strat.evaluate(series, 0))

    def test_signal_when_voters_agree(self):
        voters = [
            (_FixedVoter(Direction.CALL, 0.7), 1.0),
            (_FixedVoter(Direction.CALL, 0.6), 1.0),
            (_FixedVoter(Direction.PUT, 0.5), 0.5),
        ]
        strat = ConfluenceStrategy(voters=voters, min_voters=2, min_net_strength=0.5)
        candles = [_c(0, 10, 10, 10, 10)]
        series = CandleSeries(candles)
        signal = strat.evaluate(series, 0)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.CALL)

    def test_respects_min_voters_threshold(self):
        voters = [(_FixedVoter(Direction.CALL, 0.9), 1.0)]
        strat = ConfluenceStrategy(voters=voters, min_voters=2, min_net_strength=0.5)
        candles = [_c(0, 10, 10, 10, 10)]
        series = CandleSeries(candles)
        self.assertIsNone(strat.evaluate(series, 0))

    def test_default_confluence_runs_without_error_on_random_data(self):
        candles = random_walk_candles(150, seed=3)
        series = CandleSeries(candles)
        strat = ConfluenceStrategy()
        for i in range(strat.warmup(), len(series)):
            strat.evaluate(series, i)  # no debe lanzar excepción


if __name__ == "__main__":
    unittest.main()
