# AV2 — Classificação com NumPy

Implementação acadêmica de classificadores supervisionados para a disciplina de Inteligência Artificial Computacional da Universidade de Fortaleza (UNIFOR). Os algoritmos são implementados manualmente com NumPy e bibliotecas padrão do Python, sem pandas ou scikit-learn.

## Objetivo

Comparar k-Nearest Neighbors (kNN) com duas métricas de distância e classificadores Naive Bayes Gaussiano univariado e multivariado. A avaliação usa validação cruzada estratificada de cinco folds, métricas de classificação e tempos de treino e predição.

## Dataset

- **Fonte:** [OpenML — dataset 40701, churn](https://www.openml.org/d/40701)
- **Alvo:** `class` (`0` = não cancelou, `1` = cancelou)
- **Amostras:** 5.000
- **Atributos preditores:** 20
- **Distribuição observada:** classe 0: 4.293 (85,9%); classe 1: 707 (14,1%)

O programa consulta os metadados do OpenML e baixa o arquivo ARFF com `urllib` quando `churn.arff` não está presente. O checksum informado pelo OpenML é verificado antes de salvar o arquivo. O dataset não precisa ser versionado neste repositório.

## Classificadores

### kNN

Armazena os exemplos de treino e calcula manualmente as distâncias até cada exemplo de teste. A classe prevista é a maioria entre os `k=5` vizinhos mais próximos. São avaliadas:

- **Euclidiana:** raiz da soma dos quadrados das diferenças entre atributos.
- **Manhattan:** soma dos valores absolutos das diferenças.

### Naive Bayes Gaussiano univariado

Estima média, variância e probabilidade a priori para cada classe. Calcula a densidade normal por atributo e soma os logaritmos das probabilidades, assumindo independência condicional entre atributos. Uma variância mínima evita divisão por zero.

### Naive Bayes Gaussiano multivariado

Estima uma média e uma matriz de covariância por classe e calcula a densidade normal multivariada em domínio logarítmico. Uma regularização diagonal pequena permite inverter matrizes singulares ou mal condicionadas.

## Preparação e protocolo experimental

1. O arquivo ARFF é convertido para uma matriz NumPy com `numpy.genfromtxt`.
2. Atributos nominais são codificados manualmente seguindo a ordem dos valores declarados no ARFF.
3. Linhas sem rótulo são descartadas. Valores ausentes nas features são imputados pela média **do treino de cada fold**; a mesma média é aplicada ao teste daquele fold.
4. As features são padronizadas usando média e desvio padrão calculados somente no treino do fold, evitando vazamento de informação.
5. Os folds são estratificados e embaralhados com semente fixa (`42`). A classe positiva para precisão, recall e F1 é `1`.
6. A tabela apresenta média ± desvio padrão populacional (`numpy.std`) dos cinco folds. Os tempos medem apenas `fit` e `predict`; preparação dos dados fica fora da cronometragem.

As métricas reportadas são acurácia, precisão, recall e F1. Precisão, recall e F1 são calculados para a classe positiva `1`; denominadores nulos resultam em zero.

## Resultados de referência

Execução com `k=5`, cinco folds, semente `42`, Python 3.14.5 e NumPy 2.3.5. Os tempos variam conforme a máquina e a carga do sistema.

| Classificador | Acurácia | Precisão | Recall | F1-score | Treino (s) | Teste (s) |
|---|---:|---:|---:|---:|---:|---:|
| kNN — Euclidiana | 0,893 ± 0,008 | 0,802 ± 0,066 | 0,330 ± 0,057 | 0,464 ± 0,060 | 0,000000 ± 0,000000 | 0,186382 ± 0,000680 |
| kNN — Manhattan | **0,895 ± 0,006** | **0,867 ± 0,038** | 0,307 ± 0,063 | 0,448 ± 0,069 | 0,000000 ± 0,000000 | 0,183217 ± 0,001859 |
| Naive Bayes — univariado | 0,872 ± 0,011 | 0,549 ± 0,040 | **0,535 ± 0,041** | **0,541 ± 0,031** | 0,000604 ± 0,000493 | **0,000401 ± 0,000492** |
| Naive Bayes — multivariado | 0,872 ± 0,009 | 0,553 ± 0,035 | 0,491 ± 0,049 | 0,519 ± 0,034 | 0,001006 ± 0,000007 | 0,001037 ± 0,000051 |

### Leitura dos resultados

O Naive Bayes univariado obteve o maior F1 e o maior recall, com predição muito rápida. Neste experimento, ele teve o melhor equilíbrio entre F1 e tempo de execução. O kNN Manhattan apresentou a maior acurácia e precisão, mas também o menor recall entre as configurações avaliadas e um tempo de predição cerca de 0,19 s por fold. A diferença entre acurácia e F1 deve ser interpretada considerando o desbalanceamento do dataset: a classe positiva representa 14,1% das amostras.

O tempo de treino do kNN ficou abaixo da resolução observada em algumas medições porque o ajuste apenas armazena os dados; a maior parte do custo aparece na predição, quando as distâncias são calculadas.

## Requisitos

- Python 3.10 ou superior
- NumPy
- Acesso à internet na primeira execução ou ao usar `--refresh-data`

Nenhuma biblioteca de machine learning pronta é necessária.

## Como executar

```bash
python -m pip install -r requirements.txt
python classificacao.py
```

Para baixar novamente o dataset do OpenML:

```bash
python classificacao.py --refresh-data
```

Opções adicionais:

```text
--data CAMINHO       caminho do churn.arff
--folds N            número de folds (padrão: 5)
--neighbors N        valor de k no kNN (padrão: 5)
--seed N             semente aleatória (padrão: 42)
```

## Estrutura

```text
av2-ia-classificacao/
├── classificacao.py   # carregamento, modelos, métricas e validação cruzada
├── requirements.txt   # dependência NumPy
├── .gitignore         # cache Python e dataset baixado
└── README.md          # documentação e resultados de referência
```
