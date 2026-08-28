import random
import unittest

from pobot.data.candles import CandleSeries
from pobot.data.synthetic import random_walk_candles
from pobot.edge import breakeven_winrate, has_demonstrated_edge
from pobot.features import FeatureBuilder
from pobot.labeling import make_labels
from pobot.ml.calibration import Calibrator, reliability_curve
from pobot.ml.dataset import build_dataset, purged_split
from pobot.ml.logreg import LogisticRegression, StandardScaler, _sigmoid
from pobot.ml.sklearn_backend import SKLEARN_AVAILABLE, get_best_available_model
from pobot.strategies.ml_strategy import MLStrategy
from pobot.types import Direction


class TestStandardScaler(unittest.TestCase):
    def test_transform_gives_zero_mean_unit_ish_scale(self):
        X = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]
        scaler = StandardScaler.fit(X)
        Xs = scaler.transform(X)
        mean0 = sum(row[0] for row in Xs) / len(Xs)
        self.assertAlmostEqual(mean0, 0.0, places=9)

    def test_handles_constant_column_without_division_by_zero(self):
        X = [[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]]
        scaler = StandardScaler.fit(X)
        Xs = scaler.transform(X)
        for row in Xs:
            self.assertEqual(row[0], 0.0)


class TestLogisticRegression(unittest.TestCase):
    def test_sigmoid_bounded(self):
        self.assertAlmostEqual(_sigmoid(0.0), 0.5)
        self.assertGreater(_sigmoid(100), 0.99)
        self.assertLess(_sigmoid(-100), 0.01)

    def test_learns_linearly_separable_problem(self):
        rng = random.Random(0)
        X, y = [], []
        for _ in range(300):
            x1 = rng.gauss(0, 1)
            x2 = rng.gauss(0, 1)
            label = 1 if (2 * x1 - 3 * x2) > 0 else 0
            X.append([x1, x2])
            y.append(label)
        model = LogisticRegression(lr=0.3, epochs=100, seed=1).fit(X, y)

        correct = 0
        for xi, yi in zip(X, y):
            p = model.predict_proba_one(xi)
            pred = 1 if p >= 0.5 else 0
            correct += int(pred == yi)
        accuracy = correct / len(y)
        self.assertGreater(accuracy, 0.9)

    def test_early_stopping_uses_validation_set(self):
        rng = random.Random(2)
        X = [[rng.gauss(0, 1)] for _ in range(200)]
        y = [1 if x[0] > 0 else 0 for x in X]
        X_train, y_train = X[:150], y[:150]
        X_val, y_val = X[150:], y[150:]
        model = LogisticRegression(lr=0.5, epochs=500, patience=5, seed=3)
        model.fit(X_train, y_train, X_val, y_val)
        self.assertIsNotNone(model.weights)

    def test_raises_on_empty_dataset(self):
        with self.assertRaises(ValueError):
            LogisticRegression().fit([], [])

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            LogisticRegression().predict_proba([[1.0, 2.0]])


class TestCalibration(unittest.TestCase):
    def test_calibrator_is_monotonic(self):
        rng = random.Random(5)
        probs = [rng.random() for _ in range(500)]
        y = [1 if p + rng.gauss(0, 0.1) > 0.5 else 0 for p in probs]
        calibrator = Calibrator.fit(probs, y, n_bins=10)
        for a, b in zip(calibrator.bin_rates, calibrator.bin_rates[1:]):
            self.assertLessEqual(a, b + 1e-9)

    def test_calibrate_extrapolates_to_last_bin_above_max(self):
        calibrator = Calibrator.fit([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1], n_bins=2)
        self.assertEqual(calibrator.calibrate(0.99), calibrator.bin_rates[-1])

    def test_reliability_curve_shapes(self):
        rng = random.Random(6)
        probs = [rng.random() for _ in range(100)]
        y = [rng.randint(0, 1) for _ in range(100)]
        curve = reliability_curve(probs, y, n_bins=5)
        self.assertGreater(len(curve), 0)
        total_count = sum(b.count for b in curve)
        self.assertEqual(total_count, 100)


class TestDataset(unittest.TestCase):
    def _make_pipeline(self, n=200, horizon=3, seed=1):
        candles = random_walk_candles(n, seed=seed)
        series = CandleSeries(candles)
        rows = FeatureBuilder().build(series)
        labels = make_labels(series, horizon=horizon)
        return build_dataset(rows, labels), horizon

    def test_build_dataset_skips_invalid_and_unlabeled_rows(self):
        dataset, horizon = self._make_pipeline()
        self.assertGreater(len(dataset), 0)
        self.assertEqual(len(dataset.X), len(dataset.y))
        self.assertEqual(len(dataset.X), len(dataset.indices))
        for label in dataset.y:
            self.assertIn(label, (0, 1))

    def test_purged_split_removes_overlap_window(self):
        dataset, horizon = self._make_pipeline(n=200, horizon=5)
        split_index = 150
        train, test = purged_split(dataset, split_index, horizon)
        for idx in train.indices:
            self.assertLess(idx, split_index - horizon)
        for idx in test.indices:
            self.assertGreaterEqual(idx, split_index)
        # nada de train cae en la franja purgada [split_index - horizon, split_index)
        for idx in train.indices:
            self.assertFalse(split_index - horizon <= idx < split_index)


class TestSklearnBackend(unittest.TestCase):
    def test_get_best_available_model_never_raises(self):
        model = get_best_available_model()
        self.assertTrue(hasattr(model, "fit"))
        self.assertTrue(hasattr(model, "predict_proba"))

    def test_falls_back_to_logreg_type_when_no_sklearn(self):
        if SKLEARN_AVAILABLE:
            self.skipTest("scikit-learn está instalado en este entorno")
        model = get_best_available_model()
        self.assertIsInstance(model, LogisticRegression)


class TestMLStrategyIntegration(unittest.TestCase):
    def test_end_to_end_train_and_vote(self):
        n = 400
        horizon = 3
        candles = random_walk_candles(n, seed=42, drift=0.0002)  # pequeño drift para dar algo de señal
        series = CandleSeries(candles)
        rows = FeatureBuilder().build(series)
        labels = make_labels(series, horizon=horizon)
        dataset = build_dataset(rows, labels)

        split_index = int(len(series) * 0.7)
        train, test = purged_split(dataset, split_index, horizon)
        self.assertGreater(len(train), 20)
        self.assertGreater(len(test), 5)

        model = LogisticRegression(lr=0.2, epochs=60, seed=0).fit(train.X, train.y)
        probs_train = model.predict_proba(train.X)
        calibrator = Calibrator.fit(probs_train, train.y, n_bins=5)

        strategy = MLStrategy(model=model, calibrator=calibrator, payout=0.85, expiry_bars=horizon, min_confidence=0.0)
        votes = 0
        for i in range(strategy.warmup(), len(series) - horizon):
            vote = strategy.vote(series, i)
            if vote is not None:
                votes += 1
                self.assertIn(vote.direction, (Direction.CALL, Direction.PUT))
                self.assertGreaterEqual(vote.strength, 0.0)
                self.assertLessEqual(vote.strength, 1.0)
        self.assertGreater(votes, 0)

    def test_raises_without_model(self):
        with self.assertRaises(ValueError):
            MLStrategy(model=None)


if __name__ == "__main__":
    unittest.main()
