"""Configurações compartilhadas pelo pipeline."""
from pathlib import Path

OPENML_DATASET_ID = 40701
OPENML_METADATA_URL = f"https://www.openml.org/api/v1/json/data/{OPENML_DATASET_ID}"
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "churn.arff"
POSITIVE_LABEL = 1.0
