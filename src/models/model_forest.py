from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from notebooks.preprocess import * 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# n_estimators=100 significa que o modelo usará 100 árvores de decisão para fazer as previsões.
forest = RandomForestClassifier(n_estimators=300)

#Validação cruzada
results = cross_val_score(forest, x_train, y_train, cv=3)

forest.fit(x_train, y_train)

# Obter importâncias das features
importances = forest.feature_importances_

# Ordenar os índices das features em ordem decrescente
indices = importances.argsort()[::-1]

# Criar DataFrame com nomes das colunas (supondo que X_train foi usado no treino)
importances_df = pd.DataFrame({
    'feature': x_train.columns[indices],
    'importance': importances[indices]
})

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=importances_df, x='importance', y='feature')
plt.title("Importância das Variáveis - Random Forest")
plt.tight_layout()
plt.show()