"""Classificação do OpenML 40701 com NumPy e bibliotecas padrão do Python.

Implementa kNN (Euclidiana e Manhattan), Naive Bayes Gaussiano univariado e
multivariado, validação cruzada manual estratificada e métricas sem sklearn.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np


OPENML_DATASET_ID = 40701
OPENML_METADATA_URL = f"https://www.openml.org/api/v1/json/data/{OPENML_DATASET_ID}"
DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "churn.arff"
POSITIVE_LABEL = 1.0


def _clean_arff_token(token: str) -> str:
    """Remove espaços e aspas externas de um valor ARFF."""
    value = token.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return value.strip()


def _parse_attribute(line: str) -> tuple[str, str]:
    """Retorna nome e tipo de uma declaração @ATTRIBUTE."""
    parts = line.strip().split(None, 2)
    if len(parts) != 3 or parts[0].lower() != "@attribute":
        raise ValueError(f"Declaração ARFF inválida: {line}")
    return _clean_arff_token(parts[1]), parts[2].strip()


def _encode_arff_rows(text: str) -> tuple[list[str], list[str]]:
    """Converte a seção @DATA em linhas numéricas aceitas por genfromtxt.

    Valores nominais são codificados manualmente pela ordem da declaração
    ARFF. Colunas STRING são codificadas pela ordem em que seus valores aparecem.
    """
    attributes: list[tuple[str, str]] = []
    data_lines: list[str] = []
    in_data = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        if not in_data:
            if line.lower() == "@data":
                in_data = True
            elif line.lower().startswith("@attribute"):
                attributes.append(_parse_attribute(line))
            continue
        data_lines.append(raw_line)

    if not in_data or not attributes:
        raise ValueError("O arquivo não contém cabeçalho ARFF e seção @DATA válidos.")
    if not data_lines:
        raise ValueError("A seção @DATA do arquivo está vazia.")

    nominal_maps: list[dict[str, int] | None] = []
    string_maps: list[dict[str, int] | None] = []
    for _, attribute_type in attributes:
        type_text = attribute_type.strip()
        if type_text.startswith("{") and type_text.endswith("}"):
            domain_text = type_text[1:-1]
            domain = next(csv.reader([domain_text], skipinitialspace=True))
            domain_values = [_clean_arff_token(value) for value in domain]
            nominal_maps.append({value: index for index, value in enumerate(domain_values)})
            string_maps.append(None)
        elif type_text.lower() == "string":
            nominal_maps.append(None)
            string_maps.append({})
        else:
            nominal_maps.append(None)
            string_maps.append(None)

    numeric_lines: list[str] = []
    for row_number, row in enumerate(csv.reader(data_lines, skipinitialspace=True), start=1):
        if len(row) != len(attributes):
            raise ValueError(
                f"Linha {row_number} tem {len(row)} campos; esperados {len(attributes)}."
            )
        encoded: list[str] = []
        for column, raw_value in enumerate(row):
            value = _clean_arff_token(raw_value)
            if value == "?":
                encoded.append("?")
                continue

            category_map = nominal_maps[column]
            if category_map is not None:
                if value not in category_map:
                    raise ValueError(
                        f"Categoria {value!r} fora do domínio da coluna "
                        f"{attributes[column][0]!r}."
                    )
                encoded.append(str(category_map[value]))
                continue

            string_map = string_maps[column]
            if string_map is not None:
                if value not in string_map:
                    string_map[value] = len(string_map)
                encoded.append(str(string_map[value]))
                continue

            try:
                encoded.append(str(float(value)))
            except ValueError as error:
                raise ValueError(
                    f"Valor não numérico {value!r} na coluna "
                    f"{attributes[column][0]!r}, que não é nominal/string."
                ) from error
        numeric_lines.append(",".join(encoded))

    return [name for name, _ in attributes], numeric_lines


def parse_arff(text: str, target_name: str = "class") -> tuple[np.ndarray, np.ndarray]:
    """Lê ARFF numérico/nominal com NumPy, separa X/y e imputa faltantes."""
    attribute_names, numeric_lines = _encode_arff_rows(text)
    matches = [i for i, name in enumerate(attribute_names) if name.lower() == target_name.lower()]
    if len(matches) != 1:
        raise ValueError(f"Esperava uma coluna-alvo {target_name!r}; encontrei {len(matches)}.")

    raw = np.genfromtxt(
        io.StringIO("\n".join(numeric_lines)),
        delimiter=",",
        dtype=np.float64,
        missing_values="?",
        filling_values=np.nan,
        autostrip=True,
        ndmin=2,
    )
    if raw.ndim != 2 or raw.shape[1] != len(attribute_names):
        raise ValueError("A matriz carregada não corresponde às colunas declaradas no ARFF.")

    target_index = matches[0]
    valid_targets = ~np.isnan(raw[:, target_index])
    raw = raw[valid_targets]
    y = raw[:, target_index].copy()
    feature_indices = [i for i in range(raw.shape[1]) if i != target_index]
    X = raw[:, feature_indices].copy()

    if X.shape[0] == 0:
        raise ValueError("Não há linhas com rótulo disponível para classificação.")
    if np.any(np.isinf(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Há valores infinitos nas features ou rótulos inválidos.")
    return X, y


def _open_url(url: str, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ia-computacional-av2/1.0 (NumPy-only coursework)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def load_openml_churn(
    data_path: str | Path = DEFAULT_DATA_PATH,
    force_download: bool = False,
    timeout: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Carrega o OpenML 40701 do cache local ou baixa o ARFF via urllib."""
    path = Path(data_path)
    if path.exists() and not force_download:
        text = path.read_text(encoding="utf-8")
    else:
        metadata_bytes = _open_url(OPENML_METADATA_URL, timeout=timeout)
        metadata = json.loads(metadata_bytes.decode("utf-8"))
        description = metadata.get("data_set_description", {})
        if str(description.get("id")) != str(OPENML_DATASET_ID):
            raise ValueError("A resposta do OpenML não corresponde ao dataset 40701.")
        download_url = description.get("url")
        if not download_url:
            raise ValueError("O OpenML não forneceu o URL de download do dataset.")
        arff_bytes = _open_url(download_url, timeout=timeout)
        expected_checksum = description.get("md5_checksum")
        if expected_checksum:
            actual_checksum = hashlib.md5(arff_bytes).hexdigest()
            if actual_checksum.lower() != expected_checksum.lower():
                raise ValueError("O checksum do arquivo baixado não confere com os metadados OpenML.")
        text = arff_bytes.decode("utf-8-sig")
        if "@data" not in text.lower():
            raise ValueError("O arquivo recebido do OpenML não parece ser um ARFF.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(arff_bytes)

    X, y = parse_arff(text, target_name="class")
    if X.shape[0] <= 1000 or X.shape[1] <= 10:
        raise ValueError(
            f"O dataset não atende ao mínimo do projeto: X.shape={X.shape}; "
            "são exigidos mais de 1000 linhas e mais de 10 atributos."
        )
    return X, y


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


def k_fold_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    model: object | None = None,
    random_state: int = 42,
    positive_label: float = POSITIVE_LABEL,
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

        fit_start = time.time()
        fold_model.fit(X_train_scaled, y_train)
        train_seconds = time.time() - fit_start

        predict_start = time.time()
        predictions = fold_model.predict(X_test_scaled)
        test_seconds = time.time() - predict_start

        metrics = classification_metrics(y_test, predictions, positive_label)
        metrics["train_seconds"] = float(train_seconds)
        metrics["test_seconds"] = float(test_seconds)
        metrics["fold"] = float(fold_number + 1)
        fold_results.append(metrics)

    metric_names = ("accuracy", "precision", "recall", "f1", "train_seconds", "test_seconds")
    summary = {
        name: {
            "mean": float(np.mean([row[name] for row in fold_results])),
            "std": float(np.std([row[name] for row in fold_results])),
        }
        for name in metric_names
    }
    return {"folds": fold_results, "summary": summary}


def _format_mean_std(summary: dict[str, float], seconds: bool = False) -> str:
    mean = summary["mean"]
    std = summary["std"]
    if seconds:
        return f"{mean:.6f} ± {std:.6f}"
    return f"{mean:.3f} ± {std:.3f}"


def _run_smoke_check() -> None:
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


def print_table(results: list[tuple[str, dict[str, object]]]) -> None:
    """Imprime a Tabela 1 com média ± desvio padrão em cada coluna."""
    headers = (
        "Classificador",
        "Acurácia",
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


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Classificação NumPy-only do OpenML 40701.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Caminho do churn.arff.")
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Baixa novamente o ARFF do OpenML mesmo quando há cópia local.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Número de folds (padrão: 5).")
    parser.add_argument("--neighbors", type=int, default=5, help="k do kNN (padrão: 5).")
    parser.add_argument("--seed", type=int, default=42, help="Semente da validação cruzada.")
    args = parser.parse_args()

    _run_smoke_check()
    X, y = load_openml_churn(args.data, force_download=args.refresh_data)
    classes, counts = np.unique(y, return_counts=True)
    if not np.any(classes == POSITIVE_LABEL):
        raise ValueError(f"Rótulo positivo {POSITIVE_LABEL} não existe no dataset: {classes}.")

    print(f"OpenML dataset ID: {OPENML_DATASET_ID} (churn)")
    print(f"Shape de X: {X.shape[0]} instâncias × {X.shape[1]} atributos")
    print("Distribuição de classes: " + ", ".join(
        f"{label:g}: {count} ({count / y.size:.1%})" for label, count in zip(classes, counts)
    ))
    print(f"Validação cruzada: {args.folds}-fold estratificada; kNN k={args.neighbors}; seed={args.seed}")
    print("A padronização é ajustada apenas no treino de cada fold; tempos incluem somente fit/predict.")
    print("Self-check: split 4x2, nulos por fold, fit/predict dos quatro modelos e variância zero: OK")
    results = run_experiments(X, y, folds=args.folds, neighbors=args.neighbors, random_state=args.seed)
    print("\nTabela 1 — Média ± desvio padrão entre folds")
    print_table(results)


if __name__ == "__main__":
    main()
