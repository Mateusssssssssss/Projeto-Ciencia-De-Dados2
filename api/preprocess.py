import numpy as np
from api.model import *


def preprocess_input(data: InputData):
    """
    Recebe os dados de entrada do usuário (via API),
    aplica os encoders e transforma em vetor numpy para predição.
    """
    encoded = []

    # Converte o objeto Pydantic para dicionário
    data_dict = data.model_dump()

    for col in data_dict:
        val = data_dict[col]

        if col in encoders:
            enc = encoders[col]
            try:
                encoded_val = enc.transform([val])[0]
            except:
                # Se o valor não for reconhecido, trata como a última classe
                encoded_val = len(enc.classes_)
        else:
            encoded_val = val  # caso não tenha encoder (incomum aqui)

        encoded.append(encoded_val)

    return np.array([encoded], dtype=float)