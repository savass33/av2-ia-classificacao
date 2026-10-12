"""Contrato estrutural comum aos estimadores usados na avaliação."""
from typing import Protocol

import numpy as np


class Classifier(Protocol):
    """Interface mínima consumida pelo pipeline de validação cruzada."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Classifier": ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...
