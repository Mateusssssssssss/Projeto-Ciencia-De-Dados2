import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from keras.api.models import Sequential
from keras.api.layers import Dense
from sklearn.compose import make_column_transformer
from keras.api.utils import to_categorical


#Ler o CSV
dados = pd.read_csv('soybean.csv')
print(dados.head())
#Contar valores nulos
null = dados.isnull().sum()
print(null)
#Verificar quantidade de classes
qtd_class = dados['class'].nunique()
print(f'Quantidade de classes: {qtd_class}')
#Matrix
previsores = dados.iloc[:, 0:35]
classe = dados.iloc[:, 35]
print(classe)

#Conversão de variáveis categóricas em uma representação binária
#sparse_output=False: Garante que a saída não seja uma matriz esparsa, ou seja, ela será uma matriz densa (normal).
#remainder='passthrough': Isso significa que qualquer coluna não especificada para transformação será deixada como está (passada para a próxima etapa sem transformação).
onehotencoder = make_column_transformer((OneHotEncoder(categories='auto',sparse_output=False), previsores.columns), remainder='passthrough')

#Aprende e aplica a transformação de one-hot encoding nas colunas categóricas de previsores.
previsores_encoder  = onehotencoder.fit_transform(previsores)

#usada para codificar variáveis categóricas em números inteiros
labelencoder = LabelEncoder()
classe_encoded = labelencoder.fit_transform(classe)

# converte os números inteiros gerados pelo LabelEncoder em um formato one-hot.
classe_encoded = to_categorical(classe_encoded, num_classes=qtd_class)

#Divisão entre treino e teste(treinamento com 70% e teste com 30%)
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(previsores_encoder,
                                                                   classe_encoded,
                                                                   test_size=0.3,
                                                                   random_state=1
                                                                   )

#StandardScaler(): padroniza os dados, transformando-os para ter média 0 e desvio padrão 1. 
# Isso melhora o desempenho de modelos de machine learning que são sensíveis a escalas diferentes, 
# como SVM e redes neurais. 
# Padronização z-score
sc = StandardScaler()

#fit_transform(x_treinamento) faz duas coisas ao mesmo tempo.
#fit(): Calcula a média e o desvio padrão das colunas do x_treinamento.
#transform(): Usa a fórmula do Z-score para transformar os dados.
#Padronização dos dados
x_treinamento = sc.fit_transform(x_treinamento)
x_teste = sc.transform(x_teste)

#Classificador da rede neural
classifier = Sequential()

#Camada de ativação
classifier.add(Dense(units=64, kernel_initializer='uniform', activation='relu', input_dim= previsores_encoder.shape[1]))

# Adiciona outra camada totalmente conectada com 64 neurônios.
classifier.add(Dense(units=64, kernel_initializer='uniform', activation='relu'))

# função de ativação Softmax é muito eficaz em problemas de classificação multiclasse porque ela transforma os valores de saída da rede neural em probabilidades.
classifier.add(Dense(units=qtd_class, kernel_initializer='uniform', activation='softmax'))

#Configuração de parametros da rede neural(adam= algoritmo para atualizar os pesos e loss= cálcular erro)
# O Adam: (Adaptive Moment Estimation) é um dos otimizadores mais populares e eficientes, 
# pois ajusta automaticamente a taxa de aprendizado com base nas primeiras e segundas derivadas da função de perda. 
# Ele combina as vantagens de outros otimizadores, como o SGD (gradiente descendente estocástico) e o RMSprop.
# loss=categorical_crossentrop: é a função de perda usada para classificação multiclasse. Ela é usada quando as saídas são codificadas de forma one-hot (ou seja, cada classe é representada por um vetor onde apenas uma posição é 1 e as outras são 0).
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# accuracy: 0.9978 - loss: 0.0016 - val_accuracy: 0.9610 - val_loss: 0.4117

# Treinamento, dividindo a base de treinamento em  uma porção para validação(validation_data)
classifier.fit(x_treinamento, y_treinamento, epochs=1000, validation_data=(x_teste, y_teste))


# Previsão
previsoes = classifier.predict(x_teste)
previsoes = np.argmax(previsoes, axis=1)  # Convertendo previsões para classe única

# Convertendo y_teste de volta para as classes (sem one-hot)
y_teste_classes = np.argmax(y_teste, axis=1)

# Matriz de confusão
confusao = confusion_matrix(y_teste_classes, previsoes)
print(f'Confusão: {confusao}')

# Calculando o recall
# Usando 'macro' para média equilibrada
#o recall é importante porque indica a capacidade do modelo de identificar corretamente todos os casos positivos
recall = recall_score(y_teste_classes, previsoes, average='macro')  
print(f'Recall: {recall}')

# 'weighted' leva em consideração o desbalanceamento das classes
#Calcula a média ponderada de F1-Score, levando em consideração o número de amostras de cada classe.
f1 = f1_score(y_teste_classes, previsoes, average='weighted')  
print(f'F1-Score: {f1}')