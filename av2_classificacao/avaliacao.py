"""Métricas, validação cruzada e apresentação das experiências."""
import copy
import time

import numpy as np

from .modelos import (
    Classifier,
    KNNClassifier,
    MajorityClassifier,
    NaiveBayesMultivariate,
    NaiveBayesUnivariate,
)
from .preparacao import _standardize_fold

POSITIVE_LABEL = 1.0


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
    true_negative = int(np.sum((y_true != positive_label) & (y_pred != positive_label)))
    accuracy = float(np.mean(y_true == y_pred))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else 0.0
    )
    balanced_accuracy = (recall + specificity) / 2.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "tn": float(true_negative),
        "fp": float(false_positive),
        "fn": float(false_negative),
        "tp": float(true_positive),
    }


def k_fold_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    model: Classifier | None = None,
    random_state: int = 42,
    positive_label: float = POSITIVE_LABEL,
    timing_repeats: int = 5,
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
    if timing_repeats < 1:
        raise ValueError("timing_repeats deve ser pelo menos 1.")

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

        train_durations = []
        test_durations = []
        predictions = None
        for _ in range(timing_repeats):
            fit_start = time.perf_counter()
            fold_model.fit(X_train_scaled, y_train)
            train_durations.append(time.perf_counter() - fit_start)

            predict_start = time.perf_counter()
            predictions = fold_model.predict(X_test_scaled)
            test_durations.append(time.perf_counter() - predict_start)
        train_seconds = float(np.mean(train_durations))
        test_seconds = float(np.mean(test_durations))

        metrics = classification_metrics(y_test, predictions, positive_label)
        metrics["train_seconds"] = float(train_seconds)
        metrics["test_seconds"] = float(test_seconds)
        metrics["fold"] = float(fold_number + 1)
        fold_results.append(metrics)

    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "train_seconds",
        "test_seconds",
    )
    summary = {
        name: {
            "mean": float(np.mean([row[name] for row in fold_results])),
            "std": float(np.std([row[name] for row in fold_results])),
        }
        for name in metric_names
    }
    confusion_matrix = {
        key: int(sum(row[key] for row in fold_results)) for key in ("tn", "fp", "fn", "tp")
    }
    return {"folds": fold_results, "summary": summary, "confusion_matrix": confusion_matrix}


def _format_mean_std(summary: dict[str, float], seconds: bool = False) -> str:
    mean = summary["mean"]
    std = summary["std"]
    if seconds:
        return f"{mean:.6f} ± {std:.6f}"
    return f"{mean:.3f} ± {std:.3f}"


def print_table(results: list[tuple[str, dict[str, object]]]) -> None:
    """Imprime a Tabela 1 com média ± desvio padrão em cada coluna."""
    headers = (
        "Classificador",
        "Acurácia",
        "Acurácia balanc.",
        "Precisão",
        "Recall",
        "F1-score",
        "Treino (s)",
        "Teste (s)",
    )
    rows = []
    for name, result in results:
        summary = result["summary"]
        rows.append(
            [
                name,
                _format_mean_std(summary["accuracy"]),
                _format_mean_std(summary["balanced_accuracy"]),
                _format_mean_std(summary["precision"]),
                _format_mean_std(summary["recall"]),
                _format_mean_std(summary["f1"]),
                _format_mean_std(summary["train_seconds"], seconds=True),
                _format_mean_std(summary["test_seconds"], seconds=True),
            ]
        )
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def print_confusion_matrices(results: list[tuple[str, dict[str, object]]]) -> None:
    """Imprime contagens agregadas dos cinco folds para cada classificador."""
    print("Matrizes de confusão agregadas (linhas=real; colunas=previsto):")
    print("Classificador | TN | FP | FN | TP")
    for name, result in results:
        counts = result["confusion_matrix"]
        print(
            f"{name} | {counts['tn']} | {counts['fp']} | "
            f"{counts['fn']} | {counts['tp']}"
        )


def run_experiments(
    X: np.ndarray,
    y: np.ndarray,
    folds: int = 5,
    neighbors: int = 5,
    random_state: int = 42,
    timing_repeats: int = 5,
) -> list[tuple[str, dict[str, object]]]:
    """Executa baseline e as quatro configurações de classificação do projeto."""
    configurations = (
        ("Baseline majoritário", MajorityClassifier()),
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
            timing_repeats=timing_repeats,
        )
        results.append((name, result))
    return results
