"""Interface de linha de comando que conecta as etapas do pipeline."""
import argparse
import sys
from pathlib import Path

import numpy as np

from .config import DEFAULT_DATA_PATH, OPENML_DATASET_ID, POSITIVE_LABEL
from .data.openml import load_openml_churn
from .evaluation.reporting import print_table
from .experiments import run_experiments
from .self_check import run_self_check


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Classificação NumPy-only do OpenML 40701.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Caminho do churn.arff.")
    parser.add_argument("--refresh-data", action="store_true", help="Baixa novamente o ARFF do OpenML mesmo quando há cópia local.")
    parser.add_argument("--folds", type=int, default=5, help="Número de folds (padrão: 5).")
    parser.add_argument("--neighbors", type=int, default=5, help="k do kNN (padrão: 5).")
    parser.add_argument("--seed", type=int, default=42, help="Semente da validação cruzada.")
    args = parser.parse_args()

    run_self_check()
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
