from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix)

from notebooks.preprocess import *
from src.utils.predicts.predict_svc import *
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score

def metrics(y_true, pred_labels, pred_proba):
    """
    Avalia o desempenho de um modelo de classificação multiclasse com foco em dados desbalanceados.

    Parâmetros:
    - y_true: valores reais (classes verdadeiras).
    - y_pred_labels: rótulos de classes preditos (inteiros, de 0 a 18).
    - y_pred_proba: probabilidades preditas (shape: [n amostras, n_classes]).

    Exibe:
    - AUPRC (Área sob a Curva de Precisão-Recall)
    - Classification Report (macro)
    - Matriz de Confusão
    """

    # AUPRC Macro: ideal para multiclasse desbalanceado
    # O AUPRC macro avalia a qualidade das probabilidades para cada classe, e ainda dá o mesmo peso a cada classe, 
    # independentemente do número de exemplos.
    auprc = average_precision_score(y_true, pred_proba, average='macro')
    print(f"AUPRC (macro): {auprc}")

    # Classification report - macro = avalia cada classe igualmente
    print('Relatório de Classificação (macro):')
    print(classification_report(y_true, pred_labels, digits=3))

    # Confusion Matrix
    print('Matriz de Confusão:')
    print(confusion_matrix(y_true, pred_labels))

# Metricas para o modelo xgboost.
metrics(y_test, pred_labels_svc, pred_proba_svc)