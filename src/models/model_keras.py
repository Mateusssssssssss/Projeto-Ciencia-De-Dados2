from notebooks.preprocess_keras import * 
from keras.api.models import Sequential
from keras.api.layers import Dense




#Classificador da rede neural
classifier = Sequential()

#Camada de ativação
classifier.add(Dense(units=64, kernel_initializer='uniform', activation='relu', input_dim= previsores_encoder.shape[1])) # Numero de colunas (previsores_encoder.shape[1])

# Adiciona outra camada totalmente conectada com 64 neurônios.
classifier.add(Dense(units=64, kernel_initializer='uniform', activation='relu'))

# Adiciona outra camada totalmente conectada com 64 neurônios.
classifier.add(Dense(units=64, kernel_initializer='uniform', activation='relu'))

# função de ativação Softmax é muito eficaz em problemas de classificação multiclasse porque ela transforma os valores de saída da rede neural em probabilidades.
classifier.add(Dense(units=qtd_class, kernel_initializer='uniform', activation='softmax'))

#Configuração de parametros da rede neural(adam= algoritmo para atualizar os pesos e loss= cálcular erro)
# O Adam: (Adaptive Moment Estimation) é um dos otimizadores mais populares e eficientes, 
# pois ajusta automaticamente a taxa de aprendizado com base nas primeiras e segundas derivadas da função de perda. 
# Ele combina as vantagens de outros otimizadores, como o SGD (gradiente descendente estocástico) e o RMSprop.
# loss=categorical_crossentrop: é a função de perda usada para classificação multiclasse. Ela é usada quando as saídas são codificadas de forma one-hot (ou seja, cada classe é representada por um vetor onde apenas uma posição é 1 e as outras são 0).
classifier.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
# accuracy: 0.9978 - loss: 0.0016 - val_accuracy: 0.9610 - val_loss: 0.4117

# Treinamento, dividindo a base de treinamento em  uma porção para validação(validation_data)
classifier.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test))