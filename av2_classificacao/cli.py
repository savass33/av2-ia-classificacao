"""Comando de linha que executa a validação e imprime os resultados."""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np

from .avaliacao import POSITIVE_LABEL, print_confusion_matrices, print_table, run_experiments
from .dados import DEFAULT_DATA_PATH, OPENML_DATASET_ID, load_openml_churn
from .modelos import KNNClassifier, MajorityClassifier, NaiveBayesMultivariate, NaiveBayesUnivariate
from .preparacao import _standardize_fold, train_test_split

EXCLUDED_ABLATION_FEATURES = ("phone_number", "state", "area_code")

def run_self_check() -> None:
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
        MajorityClassifier(),
        NaiveBayesUnivariate(),
        NaiveBayesMultivariate(),
    )
    for prototype in smoke_models:
        fitted = copy.deepcopy(prototype).fit(constant_feature, y)
        predictions = fitted.predict(constant_feature[:2])
        if predictions.shape != (2,) or not np.all(np.isfinite(predictions)):
            raise AssertionError(f"Self-check falhou em {type(prototype).__name__}.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Classificação NumPy-only do OpenML 40701.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Caminho do churn.arff.")
    parser.add_argument("--refresh-data", action="store_true", help="Baixa novamente o ARFF do OpenML mesmo quando há cópia local.")
    parser.add_argument("--folds", type=int, default=5, help="Número de folds (padrão: 5).")
    parser.add_argument("--neighbors", type=int, default=5, help="k do kNN (padrão: 5).")
    parser.add_argument("--seed", type=int, default=42, help="Semente da validação cruzada.")
    parser.add_argument(
        "--timing-repeats", type=int, default=5,
        help="Repetições de fit/predict por fold para estimar tempos (padrão: 5).",
    )
    args = parser.parse_args()

    run_self_check()
    X, y = load_openml_churn(args.data, force_download=args.refresh_data)
    X_reduced, y_reduced = load_openml_churn(
        args.data,
        exclude_features=EXCLUDED_ABLATION_FEATURES,
    )
    if not np.array_equal(y, y_reduced):
        raise AssertionError("A filtragem de atributos alterou a ordem ou os rótulos do dataset.")
    classes, counts = np.unique(y, return_counts=True)
    if not np.any(classes == POSITIVE_LABEL):
        raise ValueError(f"Rótulo positivo {POSITIVE_LABEL} não existe no dataset: {classes}.")

    print(f"OpenML dataset ID: {OPENML_DATASET_ID} (churn)")
    print(f"Shape de X: {X.shape[0]} instâncias × {X.shape[1]} atributos")
    print("Distribuição de classes: " + ", ".join(
        f"{label:g}: {count} ({count / y.size:.1%})" for label, count in zip(classes, counts)
    ))
    print(f"Validação cruzada: {args.folds}-fold estratificada; kNN k={args.neighbors}; seed={args.seed}")
    print("A padronização é ajustada apenas no treino de cada fold.")
    print(
        f"Tempos: média de {args.timing_repeats} medições por fold com "
        "time.perf_counter(); são estimativas dependentes da máquina."
    )
    print("Self-check: split 4x2, nulos por fold, fit/predict dos modelos e variância zero: OK")
    results = run_experiments(
        X, y, folds=args.folds, neighbors=args.neighbors, random_state=args.seed,
        timing_repeats=args.timing_repeats,
    )
    print("\nTabela 1A — Dataset original; média ± desvio padrão entre folds")
    print_table(results)
    print_confusion_matrices(results)

    reduced_results = run_experiments(
        X_reduced, y_reduced, folds=args.folds, neighbors=args.neighbors,
        random_state=args.seed, timing_repeats=args.timing_repeats,
    )
    print(
        "\nTabela 1B — Ablação sem phone_number, state e area_code "
        "(mesmos folds); média ± desvio padrão entre folds"
    )
    print(f"Atributos preditores: {X_reduced.shape[1]}")
    print_table(reduced_results)
    print_confusion_matrices(reduced_results)
    print(
        "\nNota: state e area_code foram removidos nesta ablação porque a codificação "
        "numérica atual não representa distâncias entre categorias. Isso não demonstra "
        "que localização seja irrelevante."
    )
