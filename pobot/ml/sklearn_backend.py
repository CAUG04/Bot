"""Backend ML opcional basado en scikit-learn, con degradación automática.

Si `scikit-learn` no está instalado (no es una dependencia obligatoria del
proyecto), `get_best_available_model()` cae a `LogisticRegression` puro sin
romper nada. `HistGradientBoostingClassifier` se elige por ser robusto a
features en distintas escalas y no requerir tuning fino para un baseline.
"""

from __future__ import annotations

try:
    from sklearn.ensemble import HistGradientBoostingClassifier

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - depende del entorno del usuario
    SKLEARN_AVAILABLE = False


class SklearnModel:
    """Envoltorio con la misma interfaz que `pobot.ml.logreg.LogisticRegression`
    (`fit(X, y)`, `predict_proba(X) -> list[float]`)."""

    def __init__(self, **kwargs):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn no está instalado; usa pobot.ml.logreg.LogisticRegression")
        params = {"max_depth": 4, "max_iter": 150, "l2_regularization": 0.1, "random_state": 0}
        params.update(kwargs)
        self.model = HistGradientBoostingClassifier(**params)

    def fit(self, X: list[list[float]], y: list[int]) -> "SklearnModel":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        return [row[1] for row in self.model.predict_proba(X)]

    def predict_proba_one(self, x: list[float]) -> float:
        return self.predict_proba([x])[0]


def get_best_available_model(**kwargs):
    """Devuelve el mejor backend disponible en el entorno actual.

    Preferencia: HistGradientBoostingClassifier (sklearn) si está instalado,
    si no, regresión logística en Python puro (siempre disponible).
    """
    if SKLEARN_AVAILABLE:
        return SklearnModel(**kwargs)
    from pobot.ml.logreg import LogisticRegression

    return LogisticRegression()
