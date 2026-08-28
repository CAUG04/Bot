"""Regresión logística en Python puro (sin numpy): el modelo siempre
disponible, sin depender de que el usuario pueda instalar dependencias.

Entrenada con descenso de gradiente estocástico (SGD) + regularización L2 y
early stopping sobre un conjunto de validación (si se provee), para no
sobreajustar con pocas muestras, que es el régimen típico de un backtest.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-min(z, 700))
        return 1.0 / (1.0 + ez)
    ez = math.exp(max(z, -700))
    return ez / (1.0 + ez)


@dataclass
class StandardScaler:
    means: list[float]
    stds: list[float]

    @classmethod
    def fit(cls, X: list[list[float]]) -> "StandardScaler":
        n = len(X)
        d = len(X[0])
        means = [sum(row[j] for row in X) / n for j in range(d)]
        stds = []
        for j in range(d):
            variance = sum((row[j] - means[j]) ** 2 for row in X) / n
            stds.append(variance**0.5 if variance > 1e-12 else 1.0)
        return cls(means=means, stds=stds)

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        return [[(row[j] - self.means[j]) / self.stds[j] for j in range(len(row))] for row in X]


class LogisticRegression:
    """Clasificador binario y=1 (CALL) vs y=0 (PUT)."""

    def __init__(
        self,
        lr: float = 0.1,
        l2: float = 1e-3,
        epochs: int = 300,
        patience: int = 15,
        seed: int | None = 0,
    ):
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.scaler: StandardScaler | None = None
        self.weights: list[float] | None = None
        self.bias: float = 0.0

    def fit(
        self,
        X: list[list[float]],
        y: list[int],
        X_val: list[list[float]] | None = None,
        y_val: list[int] | None = None,
    ) -> "LogisticRegression":
        if len(X) == 0:
            raise ValueError("no se puede entrenar con un dataset vacío")
        self.scaler = StandardScaler.fit(X)
        Xs = self.scaler.transform(X)
        n, d = len(Xs), len(Xs[0])
        rng = random.Random(self.seed)

        w = [0.0] * d
        b = 0.0
        use_early_stopping = X_val is not None and y_val is not None and len(X_val) > 0
        Xs_val = self.scaler.transform(X_val) if use_early_stopping else None

        best_loss = float("inf")
        best_w, best_b = list(w), b
        no_improve = 0

        for _ in range(self.epochs):
            order = list(range(n))
            rng.shuffle(order)
            for i in order:
                xi, yi = Xs[i], y[i]
                z = b + sum(w[j] * xi[j] for j in range(d))
                p = _sigmoid(z)
                grad = p - yi
                for j in range(d):
                    w[j] -= self.lr * (grad * xi[j] + self.l2 * w[j])
                b -= self.lr * grad

            if use_early_stopping:
                loss = self._log_loss(Xs_val, y_val, w, b)
                if loss < best_loss - 1e-6:
                    best_loss = loss
                    best_w, best_b = list(w), b
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        break

        self.weights, self.bias = (best_w, best_b) if use_early_stopping else (w, b)
        return self

    @staticmethod
    def _log_loss(Xs: list[list[float]], y: list[int], w: list[float], b: float) -> float:
        eps = 1e-12
        total = 0.0
        for xi, yi in zip(Xs, y):
            z = b + sum(w[j] * xi[j] for j in range(len(xi)))
            p = min(max(_sigmoid(z), eps), 1 - eps)
            total += -(yi * math.log(p) + (1 - yi) * math.log(1 - p))
        return total / len(y)

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        if self.weights is None or self.scaler is None:
            raise RuntimeError("el modelo no está entrenado")
        Xs = self.scaler.transform(X)
        return [_sigmoid(self.bias + sum(self.weights[j] * row[j] for j in range(len(row)))) for row in Xs]

    def predict_proba_one(self, x: list[float]) -> float:
        return self.predict_proba([x])[0]

    def to_dict(self) -> dict:
        if self.weights is None or self.scaler is None:
            raise RuntimeError("el modelo no está entrenado")
        return {
            "type": "logreg",
            "weights": self.weights,
            "bias": self.bias,
            "scaler_means": self.scaler.means,
            "scaler_stds": self.scaler.stds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LogisticRegression":
        model = cls()
        model.weights = list(data["weights"])
        model.bias = float(data["bias"])
        model.scaler = StandardScaler(means=list(data["scaler_means"]), stds=list(data["scaler_stds"]))
        return model
