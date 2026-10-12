"""Configuração e execução das experiências comparativas."""
import numpy as np

from .config import POSITIVE_LABEL
from .evaluation.cross_validation import k_fold_cross_validation
from .models.knn import KNNClassifier
from .models.naive_bayes import NaiveBayesMultivariate, NaiveBayesUnivariate


def run_experiments(
    X: np.ndarray,
    y: np.ndarray,
    folds: int = 5,
    neighbors: int = 5,
    random_state: int = 42,
) -> list[tuple[str, dict[str, object]]]:
    """Executa as quatro configurações de classificação exigidas no projeto."""
    configurations = (
        (f"kNN (Euclidiana, k={neighbors})", KNNClassifier(neighbors, "euclidean")),
        (f"kNN (Manhattan, k={neighbors})", KNNClassifier(neighbors, "manhattan")),
        ("Naive Bayes Gaussiano (univariado)", NaiveBayesUnivariate()),
        ("Naive Bayes Gaussiano (multivariado)", NaiveBayesMultivariate()),
    )
    results = []
    for name, estimator in configurations:
        result = k_fold_cross_validation(
            X,
            y,
            k=folds,
            model=estimator,
            random_state=random_state,
            positive_label=POSITIVE_LABEL,
        )
        results.append((name, result))
    return results
