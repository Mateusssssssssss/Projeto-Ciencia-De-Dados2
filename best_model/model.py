import joblib
from sklearn.model_selection import cross_val_score
from notebooks.eda import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split



# Matrix
previsores = dados.iloc[:, 0:35]
classe = dados.iloc[:, 35]

x_train, x_test, y_train, y_test = train_test_split(previsores, classe, test_size=0.4, random_state=1)
# n_estimators=100 significa que o modelo usará 100 árvores de decisão para fazer as previsões.
forest = RandomForestClassifier(n_estimators=300)

#Validação cruzada
results = cross_val_score(forest, x_train, y_train, cv=3)

forest.fit(x_train, y_train)

# Salvar o pipeline completo
joblib.dump(forest, 'pipeline_soja.pkl')