
# Diagnóstico de Doenças em Soja com Rede Neural

Este projeto utiliza aprendizado de máquina para diagnosticar doenças em soja com base em um conjunto de dados que contém várias características relacionadas à planta. A análise é feita com a utilização de uma rede neural artificial.

## Bibliotecas e Ferramentas Utilizadas

- **Pandas**: Para manipulação e análise de dados.
- **Numpy**: Para operações matemáticas e manipulação de arrays.
- **Scikit-learn**: Para pré-processamento de dados, como Label Encoding e One-Hot Encoding, e divisão de dados em treino e teste.
- **Keras**: Para construir e treinar a rede neural.
- **Matplotlib**: Para visualização de gráficos e métricas.

## Passos do Processo

### 1. Leitura do Dataset
O conjunto de dados foi carregado a partir de um arquivo CSV contendo informações sobre soja.

```python
dados = pd.read_csv('soybean.csv')
```

### 2. Análise de Dados
O código realiza a verificação de valores nulos no dataset e calcula a quantidade de classes diferentes presentes no campo 'class'.

```python
null = dados.isnull().sum()
qtd_class = dados['class'].nunique()
```

### 3. Pré-processamento dos Dados
- **One-Hot Encoding**: A variável 'previsores' foi transformada em uma representação binária utilizando o OneHotEncoder.
- **Label Encoding**: A coluna 'classe' foi transformada em valores inteiros, seguidos de uma codificação One-Hot.
- **Padronização**: Os dados foram normalizados com o uso do `StandardScaler` para garantir que as variáveis tenham a mesma escala.

```python
previsores_encoder  = onehotencoder.fit_transform(previsores)
classe_encoded = labelencoder.fit_transform(classe)
classe_encoded = to_categorical(classe_encoded, num_classes=qtd_class)
```

### 4. Divisão de Dados
Os dados foram divididos entre treinamento e teste (70% para treinamento e 30% para teste).

```python
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(previsores_encoder, classe_encoded, test_size=0.3, random_state=0)
```

### 5. Construção do Modelo de Rede Neural
A rede neural foi criada com camadas totalmente conectadas. Utilizou-se a função de ativação ReLU para as camadas ocultas e Softmax na camada de saída, devido à natureza do problema de classificação multiclasse.

```python
classifier = Sequential()
classifier.add(Dense(units=64, kernel_initializer='uniform', activation='relu', input_dim= previsores_encoder.shape[1]))
classifier.add(Dense(units=64, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=qtd_class, kernel_initializer='uniform', activation='softmax'))
```

### 6. Treinamento e Validação
O modelo foi treinado utilizando o algoritmo Adam, com a função de perda `categorical_crossentropy` (ideal para problemas de classificação multiclasse).

```python
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.fit(x_treinamento, y_treinamento, epochs=1000, validation_data=(x_teste, y_teste))
```

### 7. Previsão e Avaliação
O modelo fez previsões sobre os dados de teste, e a matriz de confusão foi gerada para avaliar o desempenho do modelo.

```python
previsoes = classifier.predict(x_teste)
confusao = confusion_matrix(np.argmax(y_teste,axis=1),previsoes)
```

## Resultado

O modelo alcançou uma acurácia muito alta no conjunto de teste, demonstrando ser eficaz na classificação das doenças em soja.



## Conclusão

Este projeto é um exemplo de como redes neurais podem ser aplicadas para resolver problemas práticos, como a classificação de doenças em plantas. O uso de técnicas como One-Hot Encoding, Label Encoding, e redes neurais oferece uma solução robusta para este tipo de análise.
