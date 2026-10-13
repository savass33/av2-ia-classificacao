"""Aquisição, leitura e validação do dataset OpenML."""
import csv
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import numpy as np

OPENML_DATASET_ID = 40701
OPENML_METADATA_URL = f"https://www.openml.org/api/v1/json/data/{OPENML_DATASET_ID}"
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "churn.arff"


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


def parse_arff(
    text: str,
    target_name: str = "class",
    exclude_features: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
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
    feature_names = [attribute_names[i] for i in range(raw.shape[1]) if i != target_index]
    excluded_names = {name.lower() for name in exclude_features}
    available_names = {name.lower() for name in feature_names}
    missing_names = excluded_names - available_names
    if missing_names:
        raise ValueError(
            "Atributos solicitados para remoção não encontrados: "
            + ", ".join(sorted(missing_names))
        )
    feature_indices = [
        i
        for i, name in enumerate(attribute_names)
        if i != target_index and name.lower() not in excluded_names
    ]
    if not feature_indices:
        raise ValueError("A remoção solicitada deixou o dataset sem atributos preditores.")
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
    exclude_features: tuple[str, ...] = (),
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

    X, y = parse_arff(text, target_name="class", exclude_features=exclude_features)
    if X.shape[0] <= 1000 or X.shape[1] <= 10:
        raise ValueError(
            f"O dataset não atende ao mínimo do projeto: X.shape={X.shape}; "
            "são exigidos mais de 1000 linhas e mais de 10 atributos."
        )
    return X, y
