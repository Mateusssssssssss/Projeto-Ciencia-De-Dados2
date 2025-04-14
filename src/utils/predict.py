from src.models.model_xgboost import *
import numpy as np

#Previsao

# Obter as probabilidades preditas para todas as classes
pred_proba = modelo.predict_proba(x_test)

# Previsões padrão com o maior valor de probabilidade (sem ajuste de threshold)
pred_labels = np.argmax(pred_proba, axis=1)

# Dicionário com os thresholds personalizados
custom_thresholds = {
    14: 0.16,
    18: 0.15,
    1: 0.49,
    4: 0.18,
}

# Aplica os ajustes diretamente nas previsões
for classe, threshold in custom_thresholds.items():
    prob_classe = pred_proba[:, classe]
    pred_labels[prob_classe >= threshold] = classe
