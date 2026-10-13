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
6. A tabela apresenta média ± desvio padrão populacional (`numpy.std`) dos cinco folds. A matriz de confusão soma as previsões out-of-fold.
7. `fit` e `predict` são cronometrados com `time.perf_counter()` e repetidos cinco vezes por fold; a tabela mostra a média dessas medições. Os tempos são estimativas dependentes da máquina e da carga do sistema. A preparação fica fora da cronometragem.

As métricas reportadas são acurácia, acurácia balanceada, precisão, recall e F1. Precisão, recall e F1 são calculados para a classe positiva `1`; denominadores nulos resultam em zero.

## Resultados de referência

Execução local com `k=5`, cinco folds, semente `42`, cinco repetições de tempo por fold, Python 3.14.5 e NumPy 2.5.4. Os valores de tempo são exemplos dessa máquina; rode o programa no ambiente de apresentação para obter as estimativas locais.

### Tabela 1A — Dataset original, 20 atributos

| Classificador | Acurácia | Acurácia balanceada | Precisão | Recall | F1-score | Treino (s) | Teste (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline majoritário | 0,859 ± 0,000 | 0,500 ± 0,000 | 0,000 ± 0,000 | 0,000 ± 0,000 | 0,000 ± 0,000 | 0,000040 ± 0,000002 | 0,000003 ± 0,000000 |
| kNN — Euclidiana | 0,893 ± 0,008 | 0,658 ± 0,028 | 0,802 ± 0,066 | 0,330 ± 0,057 | 0,464 ± 0,060 | 0,000005 ± 0,000002 | 0,194839 ± 0,003099 |
| kNN — Manhattan | **0,895 ± 0,006** | 0,649 ± 0,030 | **0,867 ± 0,038** | 0,307 ± 0,063 | 0,448 ± 0,069 | 0,000004 ± 0,000000 | 0,190104 ± 0,003746 |
| Naive Bayes — univariado | 0,872 ± 0,011 | **0,731 ± 0,020** | 0,549 ± 0,040 | **0,535 ± 0,041** | **0,541 ± 0,031** | 0,000715 ± 0,000024 | **0,000272 ± 0,000014** |
| Naive Bayes — multivariado | 0,872 ± 0,009 | 0,713 ± 0,023 | 0,553 ± 0,035 | 0,491 ± 0,049 | 0,519 ± 0,034 | 0,000967 ± 0,000070 | 0,001379 ± 0,000028 |

### Leitura dos resultados

O baseline que sempre prevê a classe 0 obtém 85,86% de acurácia, mas recall, F1 e acurácia balanceada de 0,000, 0,000 e 0,500. Portanto, acurácia alta isoladamente não indica boa detecção de cancelamentos. No conjunto original, o Naive Bayes univariado obteve o maior F1 (0,541) e recall (0,535), com predição rápida. O kNN Manhattan teve a maior acurácia e precisão, mas recall de 0,307. A classe positiva corresponde a apenas 14,1% dos exemplos.

Matrizes de confusão agregadas dos cinco folds (linhas = classe real; colunas = previsão), na ordem TN, FP, FN, TP:

| Classificador | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Baseline majoritário | 4.293 | 0 | 707 | 0 |
| kNN — Euclidiana | 4.234 | 59 | 474 | 233 |
| kNN — Manhattan | 4.258 | 35 | 490 | 217 |
| Naive Bayes — univariado | 3.980 | 313 | 329 | 378 |
| Naive Bayes — multivariado | 4.011 | 282 | 360 | 347 |

### Análises complementares de atributos

O programa também executa duas ablações com os mesmos folds e a mesma semente da Tabela 1A. A primeira remove `phone_number`, que tem um valor único por registro, além de `state` e `area_code`, cujos códigos numéricos não representam distâncias categóricas. A segunda remove `total_day_charge`, `total_eve_charge`, `total_night_charge` e `total_intl_charge` para medir o efeito dos pares minutos/cobrança.

| Conjunto de atributos | Atributos | kNN Euclidiana F1 | kNN Manhattan F1 | NB univariado F1 | NB multivariado F1 |
|---|---:|---:|---:|---:|---:|
| Original | 20 | 0,464 | 0,448 | **0,541** | 0,519 |
| Sem `phone_number`, `state`, `area_code` | 17 | 0,516 | **0,529** | 0,538 | 0,517 |
| Sem as quatro cobranças | 16 | 0,400 | 0,372 | 0,445 | **0,545** |

As cobranças são quase múltiplos exatos dos minutos correspondentes: correlações observadas de 0,99999995 (diurnos), 0,99999977 (vespertinos), 0,99999921 (noturnos) e 0,99999266 (internacionais). Removê-las reduz o F1 dos dois kNN e do Naive Bayes univariado, mas eleva o F1 do multivariado de 0,519 para 0,545. Essa diferença ilustra que redundância de atributos afeta os classificadores de formas distintas.

Essas ablações são exploratórias, não provam que localização seja irrelevante e não substituem o resultado principal com os 20 atributos. Uma representação categórica justa exigiria preparação apropriada a cada algoritmo; a codificação ordinal atual não é uma distância adequada para kNN. Os resultados também usam uma única semente: o desvio entre folds é descritivo e não demonstra significância estatística.

O treino do kNN é rápido porque `fit` armazena os dados; a maior parte do custo aparece na predição. Mesmo com cinco medições por fold, esses tempos são aproximados e podem variar entre máquinas.

## Requisitos

- Python 3.10 ou superior
- NumPy
- Acesso à internet na primeira execução ou ao usar `--refresh-data`

Nenhuma biblioteca de machine learning pronta é necessária.

## Como executar

```bash
python -m pip install -r requirements.txt
python -m av2_classificacao
```

O comando `python classificacao.py` continua disponível como atalho compatível.

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
--timing-repeats N   repetições de fit/predict por fold para estimar tempos (padrão: 5)
```

## Estrutura

```text
av2-ia-classificacao/
├── av2_classificacao/
│   ├── dados.py                   # download, parser ARFF, codificação, X/y
│   ├── preparacao.py              # split manual, imputação e padronização
│   ├── modelos.py                 # contrato, baseline e classificadores
│   ├── avaliacao.py               # métricas, 5-fold, tempos e tabelas
│   ├── cli.py                     # self-check e execução do pipeline
│   ├── __main__.py                # entrada: python -m av2_classificacao
│   └── __init__.py                # interface pública do pacote
├── classificacao.py               # atalho legado para execução
├── requirements.txt               # NumPy
├── .gitignore                     # cache Python e dataset baixado
└── README.md                      # documentação e resultados
```

### Organização e princípios SOLID

- **Responsabilidade única:** os cinco módulos centrais separam dados, preparação, modelos, avaliação e execução.
- **Inversão de dependência:** a validação cruzada recebe qualquer estimador que implemente o contrato estrutural `Classifier` (`fit` e `predict`), sem depender de uma classe concreta.
- **Aberto para extensão:** um novo classificador pode ser adicionado a `modelos.py` e registrado em `avaliacao.py` sem alterar a leitura dos dados.
- **Interface enxuta:** os estimadores expõem somente as operações necessárias para treino e predição.

Os módulos usam NumPy e a biblioteca padrão; não há dependência de pandas ou scikit-learn.
