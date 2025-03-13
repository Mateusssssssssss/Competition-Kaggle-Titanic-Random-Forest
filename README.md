# Titanic Survival Prediction Using Random Forest Classifier

Este projeto utiliza o **Random Forest Classifier** para prever a sobrevivência de passageiros no **Titanic**. O modelo é treinado com base em dados históricos, que incluem informações sobre os passageiros, como idade, classe do passageiro, tarifa paga, entre outras.

## Tecnologias Utilizadas
- **Python**: Linguagem de programação para análise e manipulação de dados.
- **pandas**: Biblioteca para manipulação de dados.
- **scikit-learn**: Biblioteca para criação e avaliação de modelos de machine learning.
- **seaborn** e **matplotlib**: Bibliotecas para visualização de dados.
- **LabelEncoder**: Para converter variáveis categóricas em variáveis numéricas.
- **RandomForestClassifier**: Modelo de machine learning baseado em uma floresta aleatória para classificação.

## Descrição do Código

1. **Carregamento e Exploração dos Dados**:
   - O código começa lendo o arquivo `train.csv` com os dados do Titanic.
   - Realiza a visualização inicial dos dados e exibe estatísticas, como a média e a moda da coluna `Age`.
   - A coluna `Age` é tratada para substituir valores nulos pela mediana da coluna.

2. **Pré-processamento dos Dados**:
   - A coluna `Sex` (sexo dos passageiros) é transformada em valores numéricos usando o **LabelEncoder**.
   - O conjunto de dados é dividido entre variáveis preditoras (como `Age`, `Fare`, `Pclass`, `Sex`, etc.) e a variável de resposta (`Survived`).

3. **Divisão dos Dados**:
   - O conjunto de dados é dividido em duas partes: 70% para treinamento e 30% para teste, usando a função `train_test_split`.

4. **Treinamento e Avaliação do Modelo**:
   - O **RandomForestClassifier** é treinado com os dados de treinamento (`x_treinamento` e `y_treinamento`).
   - A importância das variáveis preditoras é calculada, destacando as características mais relevantes para a previsão.

5. **Previsão e Avaliação**:
   - O modelo faz previsões no conjunto de dados de teste.
   - A **matriz de confusão** é gerada para avaliar o desempenho do modelo.
   - A **taxa de acerto** (accuracy) e a **taxa de erro** (error rate) são calculadas.

6. **Validação Cruzada**:
   - A **validação cruzada** é utilizada para avaliar a robustez do modelo, dividindo os dados em 10 partes (folds) e treinando o modelo em diferentes divisões.

7. **Predição no Conjunto de Teste**:
   - O modelo é utilizado para prever a sobrevivência dos passageiros no conjunto de dados de teste (`test.csv`).
   - Valores nulos em `Age` e `Fare` são preenchidos pela mediana de cada coluna.

8. **Resultados**:
   - As previsões de sobrevivência são obtidas para os passageiros do conjunto de teste.

2. **Baixar os dados**:
   - Certifique-se de ter os arquivos `train.csv` e `test.csv` com os dados do Titanic.
   
3. **Executar o código**:
   - O código pode ser executado diretamente em um script Python. Ele irá carregar os dados, treinar o modelo e gerar as previsões de sobrevivência.

## Conclusão

O modelo de **Random Forest Classifier** apresentou um bom desempenho na previsão de sobreviventes do Titanic. A validação cruzada ajuda a garantir que o modelo seja robusto e tenha uma boa generalização, evitando overfitting.

## Resultados

- **Média de Acurácia (Validação Cruzada)**: ~80.75%
- **Taxa de Acuracidade no Conjunto de Teste**: ~82.46%
- **Taxa de Erro no Conjunto de Teste**: ~17.54%

## Observações

- O modelo pode ser aprimorado com mais dados, ajuste de hiperparâmetros e técnicas de feature engineering.
- Pode-se experimentar outros modelos de classificação, como **SVM**, **Logistic Regression**, ou até mesmo melhorar o modelo com **Boosting**.



