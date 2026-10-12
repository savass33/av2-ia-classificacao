"""Formatação da tabela de resultados experimentais."""


def _format_mean_std(summary: dict[str, float], seconds: bool = False) -> str:
    mean = summary["mean"]
    std = summary["std"]
    if seconds:
        return f"{mean:.6f} ± {std:.6f}"
    return f"{mean:.3f} ± {std:.3f}"


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
