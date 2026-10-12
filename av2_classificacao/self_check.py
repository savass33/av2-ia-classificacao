"""Verificações rápidas de invariantes dos componentes do projeto."""
import copy

import numpy as np

from .models.knn import KNNClassifier
from .models.naive_bayes import NaiveBayesMultivariate, NaiveBayesUnivariate
from .preprocessing.fold import _standardize_fold
from .preprocessing.split import train_test_split


def run_self_check() -> None:
    """Checagens pequenas de shape, divisão, fit/predict e variância zero."""
    X = np.array([[0.0, 0.0], [0.2, 0.1], [1.0, 1.0], [0.9, 0.8]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
    if X_train.shape != (3, 2) or X_test.shape != (1, 2) or y_train.size != 3 or y_test.size != 1:
        raise AssertionError("train_test_split falhou no exemplo 4x2.")

    X_with_missing = np.array([[0.0, np.nan], [1.0, 2.0]])
    X_test_with_missing = np.array([[np.nan, 3.0]])
    prepared_train, prepared_test = _standardize_fold(X_with_missing, X_test_with_missing)
    if not np.all(np.isfinite(prepared_train)) or not np.all(np.isfinite(prepared_test)):
        raise AssertionError("A imputação manual por fold falhou no caso com valor nulo.")

    constant_feature = np.column_stack((X, np.ones(X.shape[0])))
    smoke_models = (
        KNNClassifier(k=1, metric="euclidean"),
        KNNClassifier(k=1, metric="manhattan"),
        NaiveBayesUnivariate(),
        NaiveBayesMultivariate(),
    )
    for prototype in smoke_models:
        fitted = copy.deepcopy(prototype).fit(constant_feature, y)
        predictions = fitted.predict(constant_feature[:2])
        if predictions.shape != (2,) or not np.all(np.isfinite(predictions)):
            raise AssertionError(f"Self-check falhou em {type(prototype).__name__}.")
