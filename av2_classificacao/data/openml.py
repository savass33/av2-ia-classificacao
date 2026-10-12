"""Baixa e valida o dataset OpenML, mantendo uma cópia local em cache."""
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

from ..config import DEFAULT_DATA_PATH, OPENML_DATASET_ID, OPENML_METADATA_URL
from .arff import parse_arff


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
