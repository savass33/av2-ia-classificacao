"""Aquisição e leitura de dados."""
from .openml import load_openml_churn
from .arff import parse_arff

__all__ = ["load_openml_churn", "parse_arff"]
