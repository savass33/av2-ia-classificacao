"""Classificadores Naive Bayes Gaussianos univariado e multivariado."""
import math

import numpy as np


class NaiveBayesUnivariate:
    """Naive Bayes Gaussiano: densidade normal univariada por atributo."""

    def __init__(self, variance_floor: float = 1e-9) -> None:
        self.variance_floor = float(variance_floor)
        self.classes_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.variances_: np.ndarray | None = None
        self.priors_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBayesUnivariate":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("fit espera X 2D e y 1D com o mesmo número de linhas.")
        self.classes_, counts = np.unique(y, return_counts=True)
        n_classes, n_features = self.classes_.size, X.shape[1]
        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.variances_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors_ = counts.astype(np.float64) / X.shape[0]
        for class_index, class_value in enumerate(self.classes_):
            X_class = X[y == class_value]
            self.means_[class_index] = np.mean(X_class, axis=0)
            self.variances_[class_index] = np.maximum(
                np.var(X_class, axis=0), self.variance_floor
            )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.means_ is None or self.variances_ is None:
            raise RuntimeError("Chame fit(X, y) antes de predict(X).")
        X = np.asarray(X, dtype=np.float64)
        log_probabilities = []
        for index, _ in enumerate(self.classes_):
            variance = self.variances_[index]
            difference = X - self.means_[index]
            # Log da densidade normal univariada, somado sob a hipótese de
            # independência condicional dos atributos.
            log_density = -0.5 * np.sum(
                np.log(2.0 * math.pi * variance) + difference * difference / variance,
                axis=1,
            )
            log_probabilities.append(math.log(self.priors_[index]) + log_density)
        scores = np.column_stack(log_probabilities)
        return self.classes_[np.argmax(scores, axis=1)]


class NaiveBayesMultivariate:
    """Naive Bayes com uma normal multivariada completa por classe."""

    def __init__(self, regularization: float = 1e-6) -> None:
        self.regularization = float(regularization)
        self.classes_: np.ndarray | None = None
        self.means_: list[np.ndarray] = []
        self.inverse_covariances_: list[np.ndarray] = []
        self.log_determinants_: list[float] = []
        self.priors_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBayesMultivariate":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("fit espera X 2D e y 1D com o mesmo número de linhas.")
        self.classes_, counts = np.unique(y, return_counts=True)
        self.priors_ = counts.astype(np.float64) / X.shape[0]
        self.means_ = []
        self.inverse_covariances_ = []
        self.log_determinants_ = []
        n_features = X.shape[1]

        for class_value in self.classes_:
            X_class = X[y == class_value]
            mean = np.mean(X_class, axis=0)
            if X_class.shape[0] > 1:
                covariance = np.atleast_2d(np.cov(X_class, rowvar=False, ddof=1))
            else:
                covariance = np.eye(n_features, dtype=np.float64)

            average_variance = float(np.trace(covariance) / n_features)
            if not math.isfinite(average_variance) or average_variance <= 0.0:
                average_variance = 1.0
            ridge = self.regularization * average_variance
            covariance = covariance + ridge * np.eye(n_features, dtype=np.float64)

            # Ridge evita singularidade em atributos colineares; inv/det seguem
            # a expressão da normal multivariada. slogdet cobre underflow do det.
            inverse_covariance = np.linalg.inv(covariance)
            determinant = float(np.linalg.det(covariance))
            if determinant > 0.0 and math.isfinite(determinant):
                log_determinant = math.log(determinant)
            else:
                sign, log_determinant = np.linalg.slogdet(covariance)
                if sign <= 0 or not math.isfinite(float(log_determinant)):
                    raise np.linalg.LinAlgError(
                        "Covariância não positiva definida mesmo após regularização."
                    )

            self.means_.append(mean)
            self.inverse_covariances_.append(inverse_covariance)
            self.log_determinants_.append(float(log_determinant))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.priors_ is None:
            raise RuntimeError("Chame fit(X, y) antes de predict(X).")
        X = np.asarray(X, dtype=np.float64)
        n_features = X.shape[1]
        scores = []
        for index, _ in enumerate(self.classes_):
            differences = X - self.means_[index]
            quadratic = np.einsum(
                "bi,ij,bj->b",
                differences,
                self.inverse_covariances_[index],
                differences,
            )
            log_density = -0.5 * (
                n_features * math.log(2.0 * math.pi)
                + self.log_determinants_[index]
                + quadratic
            )
            scores.append(math.log(self.priors_[index]) + log_density)
        return self.classes_[np.argmax(np.column_stack(scores), axis=1)]
