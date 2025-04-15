from notebooks.eda import qtd_class, dados
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from keras.api.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Matrix
previsores = dados.iloc[:, 0:35]
classe = dados.iloc[:, 35]


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
# Divisão dos dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(
    previsores_encoder, classe_encoded, test_size=0.3, random_state=1
)

#StandardScaler(): padroniza os dados, transformando-os para ter média 0 e desvio padrão 1. 
# Isso melhora o desempenho de modelos de machine learning que são sensíveis a escalas diferentes, 
# como SVM e redes neurais. 
# Padronização z-score
sc = StandardScaler()