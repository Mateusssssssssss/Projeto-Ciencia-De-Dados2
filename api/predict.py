import joblib
from api.preprocess import preprocess_input
import numpy as np
# Carrega modelo
pipeline = joblib.load("best_model/pipeline_soja.pkl")

def prediction(data):
    input_dict = data.model_dump()  # Converte o objeto Pydantic (InputData) em um dicionário
    entrada = preprocess_input(input_dict)  # Pré-processa os dados para o modelo
    resultado = pipeline.predict_proba(entrada)  # Gera as probabilidades de cada classe com o modelo treinado
    
    # Dicionário com os thresholds personalizados para cada tipo de doença
    custom_thresholds = {
        0: 0.16,  # Doença 0 (exemplo: 0.16 como limiar)
        1: 0.49,  # Doença 1
        2: 0.30,  # Doença 2
        14: 0.16,
        18: 0.15,
        3: 0.45,
        4: 0.18,
        # Adicione mais thresholds conforme necessário
    }

    # Obter as probabilidades preditas para todas as classes
    pred_proba_forest = resultado[0]  # Considerando que resultado é uma matriz 2D (probabilidades para todas as classes)
    
    # Inicializa a previsão como uma lista de 0s para cada classe
    pred_labels_forest = np.zeros_like(pred_proba_forest, dtype=int)

    # Aplica os ajustes diretamente nas previsões usando os limiares
    for classe, threshold in custom_thresholds.items():
        prob_classe = pred_proba_forest[classe]  # Probabilidade da classe específica
        if prob_classe >= threshold:
            pred_labels_forest[classe] = 1  # Classe predita como '1' se probabilidade maior que o limiar

    # Identifica a classe predita com base no limiar
    predicted_class = np.argmax(pred_labels_forest)  # Classe com a maior probabilidade ajustada pelos limiares

    # Exibe o resultado com base na classe predita
    return {
        "probabilidade": round(float(pred_proba_forest[predicted_class]), 3),  # Exibe a probabilidade da classe predita
        "Resultado": f"Doença {predicted_class}"  # Exibe o nome da doença predita (classe)
    }
