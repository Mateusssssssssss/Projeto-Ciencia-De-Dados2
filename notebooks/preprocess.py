from notebooks.eda import dados
from sklearn.model_selection import train_test_split

# Matrix
previsores = dados.iloc[:, 0:35]
classe = dados.iloc[:, 35]
# Divisão dos dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(
    previsores, classe, test_size=0.4, random_state=1
)