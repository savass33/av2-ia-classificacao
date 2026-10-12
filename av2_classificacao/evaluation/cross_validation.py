"""Validação cruzada estratificada e cronometragem do ajuste e predição."""
import copy
import time

import numpy as np

from ..config import POSITIVE_LABEL
from ..models.base import Classifier
from ..preprocessing.fold import _standardize_fold
from .metrics import classification_metrics


def k_fold_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    model: Classifier | None = None,
    random_state: int = 42,
    positive_label: float = POSITIVE_LABEL,
) -> dict[str, object]:
    """Validação cruzada k-fold estratificada, particionada manualmente."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if model is None:
        raise ValueError("Informe um modelo com métodos fit(X, y) e predict(X).")
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("X deve ser 2D e y 1D, com o mesmo número de amostras.")
    if k < 2 or k > X.shape[0]:
        raise ValueError("k deve estar entre 2 e o número de amostras.")

    rng = np.random.default_rng(random_state)
    fold_buckets: list[list[int]] = [[] for _ in range(k)]
    for class_value in np.unique(y):
        class_indices = rng.permutation(np.flatnonzero(y == class_value))
        for position, sample_index in enumerate(class_indices):
            fold_buckets[position % k].append(int(sample_index))
    folds = [rng.permutation(np.asarray(bucket, dtype=np.int64)) for bucket in fold_buckets]
    all_indices = np.arange(X.shape[0])

    fold_results: list[dict[str, float]] = []
    for fold_number, test_indices in enumerate(folds):
        train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        X_train_scaled, X_test_scaled = _standardize_fold(X_train, X_test)
        fold_model = copy.deepcopy(model)

        fit_start = time.time()
        fold_model.fit(X_train_scaled, y_train)
        train_seconds = time.time() - fit_start

        predict_start = time.time()
        predictions = fold_model.predict(X_test_scaled)
        test_seconds = time.time() - predict_start

        metrics = classification_metrics(y_test, predictions, positive_label)
        metrics["train_seconds"] = float(train_seconds)
        metrics["test_seconds"] = float(test_seconds)
        metrics["fold"] = float(fold_number + 1)
        fold_results.append(metrics)

    metric_names = ("accuracy", "precision", "recall", "f1", "train_seconds", "test_seconds")
    summary = {
        name: {
            "mean": float(np.mean([row[name] for row in fold_results])),
            "std": float(np.std([row[name] for row in fold_results])),
        }
        for name in metric_names
    }
    return {"folds": fold_results, "summary": summary}
