"""Classificadores NumPy para o dataset OpenML 40701."""
from .models.knn import KNNClassifier
from .models.naive_bayes import NaiveBayesMultivariate, NaiveBayesUnivariate

__all__ = ["KNNClassifier", "NaiveBayesUnivariate", "NaiveBayesMultivariate"]
