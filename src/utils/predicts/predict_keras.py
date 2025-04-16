from src.models.model_keras import *
import numpy as np
# Obter as probabilidades preditas para todas as classes
pred_proba = classifier.predict(x_test)

# Labels preditos (classe com maior probabilidade)
pred_labels = np.argmax(pred_proba, axis=1)

# Labels verdadeiros (caso estejam em one-hot)
y_true = np.argmax(y_test, axis=1)
