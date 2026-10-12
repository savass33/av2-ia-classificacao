"""Imputação e padronização aprendidas exclusivamente no fold de treino."""
import numpy as np


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
