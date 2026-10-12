"""Divisão dos dados e preparação de cada fold sem vazamento de informação."""
import math

import numpy as np


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divisão manual inspirada no script hould_out.py fornecido."""
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X e y devem ter o mesmo número de amostras.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size deve estar entre 0 e 1.")
    n_samples = X.shape[0]
    n_test = math.ceil(n_samples * test_size)
    if n_test <= 0 or n_test >= n_samples:
        raise ValueError("A divisão deve deixar pelo menos uma amostra em cada conjunto.")
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def _standardize_fold(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Imputa nulos e padroniza usando estatísticas apenas do treino do fold."""
    X_train = np.asarray(X_train, dtype=np.float64).copy()
    X_test = np.asarray(X_test, dtype=np.float64).copy()
    if np.any(np.isinf(X_train)) or np.any(np.isinf(X_test)):
        raise ValueError("As features contêm valores infinitos.")
    means = np.zeros(X_train.shape[1], dtype=np.float64)
    for column in range(X_train.shape[1]):
        observed = X_train[np.isfinite(X_train[:, column]), column]
        means[column] = float(np.mean(observed)) if observed.size else 0.0
        X_train[np.isnan(X_train[:, column]), column] = means[column]
        X_test[np.isnan(X_test[:, column]), column] = means[column]

    scales = np.std(X_train, axis=0)
    scales[scales == 0.0] = 1.0
    X_train = (X_train - means) / scales
    X_test = (X_test - means) / scales
    if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(X_test)):
        raise ValueError("Não foi possível preparar o fold com valores finitos.")
    return X_train, X_test
