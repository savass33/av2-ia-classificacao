"""Classificador kNN com cálculo manual das distâncias."""
import numpy as np


class KNNClassifier:
    """kNN com cálculo explícito das distâncias Euclidiana ou Manhattan."""

    def __init__(self, k: int = 5, metric: str = "euclidean") -> None:
        if k < 1:
            raise ValueError("k deve ser um inteiro positivo.")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric deve ser 'euclidean' ou 'manhattan'.")
        self.k = int(k)
        self.metric = metric
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("fit espera X 2D e y 1D com o mesmo número de linhas.")
        if self.k > X.shape[0]:
            raise ValueError("k não pode ser maior que o número de exemplos de treino.")
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Chame fit(X, y) antes de predict(X).")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.X_train.shape[1]:
            raise ValueError("predict recebeu quantidade de atributos incompatível.")
        predictions = []
        for sample in X:
            differences = self.X_train - sample
            if self.metric == "euclidean":
                distances = np.sqrt(np.sum(differences * differences, axis=1))
            else:
                distances = np.sum(np.abs(differences), axis=1)
            nearest_indices = np.argsort(distances)[: self.k]
            labels, counts = np.unique(self.y_train[nearest_indices], return_counts=True)
            predictions.append(labels[int(np.argmax(counts))])
        return np.asarray(predictions, dtype=self.y_train.dtype)
