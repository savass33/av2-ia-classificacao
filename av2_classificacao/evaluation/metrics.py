"""Métricas de classificação calculadas sem bibliotecas prontas."""
import numpy as np

from ..config import POSITIVE_LABEL


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: float = POSITIVE_LABEL,
) -> dict[str, float]:
    """Calcula acurácia, precisão, recall e F1 para a classe positiva."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 1 or y_true.size == 0:
        raise ValueError("y_true e y_pred devem ser vetores 1D não vazios e iguais em tamanho.")
    true_positive = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    false_positive = int(np.sum((y_true != positive_label) & (y_pred == positive_label)))
    false_negative = int(np.sum((y_true == positive_label) & (y_pred != positive_label)))
    accuracy = float(np.mean(y_true == y_pred))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
