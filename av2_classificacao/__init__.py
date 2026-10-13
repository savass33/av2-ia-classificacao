"""Projeto acadêmico de classificação com NumPy."""
from .modelos import (
    KNNClassifier,
    MajorityClassifier,
    NaiveBayesMultivariate,
    NaiveBayesUnivariate,
)

__all__ = [
    "KNNClassifier",
    "MajorityClassifier",
    "NaiveBayesUnivariate",
    "NaiveBayesMultivariate",
]
