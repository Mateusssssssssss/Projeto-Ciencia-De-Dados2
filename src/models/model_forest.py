from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from notebooks.preprocess import * 

# n_estimators=100 significa que o modelo usará 100 árvores de decisão para fazer as previsões.
forest = RandomForestClassifier(n_estimators=300)

#Validação cruzada
results = cross_val_score(forest, x_train, y_train, cv=3)

forest.fit(x_train, y_train)

