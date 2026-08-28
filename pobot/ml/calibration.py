"""Calibración de probabilidades por binning + curva de fiabilidad.

La decisión de operar depende de la PROBABILIDAD calibrada, no del score
crudo del modelo: un modelo mal calibrado puede tener buen ranking (AUC) y
aun así dar probabilidades que no reflejan el winrate real, lo que rompe
directamente el criterio de edge (`pobot.edge.has_demonstrated_edge`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Calibrator:
    """Mapea una probabilidad cruda a una probabilidad calibrada, usando
    bins monótonos por cuantil ajustados sobre un conjunto de validación."""

    bin_edges: list[float]  # borde superior (probabilidad cruda) de cada bin
    bin_rates: list[float]  # winrate empírico observado en ese bin

    @classmethod
    def fit(cls, probs: list[float], y: list[int], n_bins: int = 10) -> "Calibrator":
        if not probs:
            raise ValueError("no se puede calibrar sin datos")
        pairs = sorted(zip(probs, y), key=lambda pair: pair[0])
        n = len(pairs)
        bin_size = max(1, n // n_bins)

        edges: list[float] = []
        rates: list[float] = []
        i = 0
        while i < n:
            chunk = pairs[i : i + bin_size]
            edges.append(chunk[-1][0])
            rates.append(sum(c[1] for c in chunk) / len(chunk))
            i += bin_size

        # Fuerza monotonía no decreciente (isotonic simple): si la calibración
        # cruda ya es informativa, un bin de score más alto no debería tener
        # menor winrate empírico que el anterior.
        for k in range(1, len(rates)):
            if rates[k] < rates[k - 1]:
                rates[k] = rates[k - 1]

        return cls(bin_edges=edges, bin_rates=rates)

    def calibrate(self, p: float) -> float:
        for edge, rate in zip(self.bin_edges, self.bin_rates):
            if p <= edge:
                return rate
        return self.bin_rates[-1] if self.bin_rates else p

    def calibrate_many(self, probs: list[float]) -> list[float]:
        return [self.calibrate(p) for p in probs]

    def to_dict(self) -> dict:
        return {"bin_edges": self.bin_edges, "bin_rates": self.bin_rates}

    @classmethod
    def from_dict(cls, data: dict) -> "Calibrator":
        return cls(bin_edges=list(data["bin_edges"]), bin_rates=list(data["bin_rates"]))


@dataclass
class ReliabilityBin:
    predicted_mean: float
    empirical_rate: float
    count: int


def reliability_curve(probs: list[float], y: list[int], n_bins: int = 10) -> list[ReliabilityBin]:
    """Para diagnóstico: compara la probabilidad media predicha por bin
    contra el winrate empírico observado en ese bin. En un modelo bien
    calibrado, ambas columnas deben ser casi iguales."""
    if not probs:
        return []
    pairs = sorted(zip(probs, y), key=lambda pair: pair[0])
    n = len(pairs)
    bin_size = max(1, n // n_bins)
    out = []
    i = 0
    while i < n:
        chunk = pairs[i : i + bin_size]
        predicted_mean = sum(c[0] for c in chunk) / len(chunk)
        empirical_rate = sum(c[1] for c in chunk) / len(chunk)
        out.append(ReliabilityBin(predicted_mean, empirical_rate, len(chunk)))
        i += bin_size
    return out
