"""Divisão manual dos dados para holdout."""
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
