"""Estrategia que envuelve un modelo de ML entrenado (logreg puro o backend
de scikit-learn) + un calibrador opcional.

El umbral de confianza por defecto es el winrate de equilibrio del payout
configurado: por debajo de ese umbral, aunque el modelo "se incline" hacia
una dirección, no hay ventaja económica esperada y no se debe operar.
"""

from __future__ import annotations

from typing import Optional, Protocol

from pobot.data.candles import CandleSeries
from pobot.edge import breakeven_winrate
from pobot.features import FeatureBuilder
from pobot.ml.calibration import Calibrator
from pobot.strategies.base import IndicatorCacheMixin, Vote, VotingStrategy, register_strategy
from pobot.types import Direction


class _ProbModel(Protocol):
    def predict_proba_one(self, x: list[float]) -> float: ...


@register_strategy
class MLStrategy(IndicatorCacheMixin, VotingStrategy):
    name = "ml"

    def __init__(
        self,
        model: Optional[_ProbModel] = None,
        calibrator: Optional[Calibrator] = None,
        feature_builder: Optional[FeatureBuilder] = None,
        payout: float = 0.85,
        expiry_bars: int = 3,
        min_confidence: Optional[float] = None,
        feature_warmup: int = 60,
    ):
        min_conf = min_confidence if min_confidence is not None else breakeven_winrate(payout)
        super().__init__(expiry_bars=expiry_bars, min_confidence=min_conf)
        if model is None:
            raise ValueError("MLStrategy requiere un modelo ya entrenado (ver pobot.ml)")
        self.model = model
        self.calibrator = calibrator
        self.feature_builder = feature_builder or FeatureBuilder()
        self.payout = payout
        self.feature_warmup = feature_warmup

    def warmup(self) -> int:
        return self.feature_warmup

    def _build_cache(self, series: CandleSeries):
        return self.feature_builder.build(series)

    def vote(self, series: CandleSeries, i: int) -> Optional[Vote]:
        rows = self._get_cache(series)
        row = rows[i]
        if not row.valid:
            return None

        raw_p_call = self.model.predict_proba_one(row.as_vector())
        p_call = self.calibrator.calibrate(raw_p_call) if self.calibrator else raw_p_call

        if p_call >= 0.5:
            return Vote(Direction.CALL, p_call, [f"ML: p(CALL) calibrada = {p_call:.3f}"])
        p_put = 1.0 - p_call
        return Vote(Direction.PUT, p_put, [f"ML: p(PUT) calibrada = {p_put:.3f}"])
