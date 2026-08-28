"""Construcción de datasets X/y a partir de features y etiquetas, con splits
"purgados" para evitar fuga de información entre train y test.

Purga: como la etiqueta de la barra `t` depende de `close[t+horizon]`, una
fila de train cuyo índice cae dentro de `[split_index - horizon, split_index)`
"ve" información que se solapa temporalmente con el inicio del test. Se
descarta esa franja en vez de usarla, siguiendo la práctica estándar de
purged cross-validation en series temporales financieras.
"""

from __future__ import annotations

from dataclasses import dataclass

from pobot.features import FeatureRow
from pobot.labeling import LabelSet
from pobot.types import Direction


@dataclass
class Dataset:
    X: list[list[float]]
    y: list[int]  # 1 = CALL, 0 = PUT
    indices: list[int]  # índice de barra `t` de cada fila

    def __len__(self) -> int:
        return len(self.y)


def build_dataset(feature_rows: list[FeatureRow], labels: LabelSet) -> Dataset:
    X: list[list[float]] = []
    y: list[int] = []
    indices: list[int] = []
    for row in feature_rows:
        if not row.valid:
            continue
        direction = labels.direction[row.index]
        if direction is None:  # sin etiqueta (cola de la serie) o empate en modo refund
            continue
        X.append(row.as_vector())
        y.append(1 if direction is Direction.CALL else 0)
        indices.append(row.index)
    return Dataset(X, y, indices)


def purged_split(dataset: Dataset, split_index: int, horizon: int) -> tuple[Dataset, Dataset]:
    """Divide por índice de barra: train = índices < split_index - horizon,
    test = índices >= split_index. La franja intermedia se descarta."""
    purge_start = split_index - horizon
    train_X, train_y, train_idx = [], [], []
    test_X, test_y, test_idx = [], [], []
    for x, yy, idx in zip(dataset.X, dataset.y, dataset.indices):
        if idx < purge_start:
            train_X.append(x)
            train_y.append(yy)
            train_idx.append(idx)
        elif idx >= split_index:
            test_X.append(x)
            test_y.append(yy)
            test_idx.append(idx)
    return Dataset(train_X, train_y, train_idx), Dataset(test_X, test_y, test_idx)
