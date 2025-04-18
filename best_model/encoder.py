import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Carrega seu dataset (ajuste o caminho se necessário)
def load_data():
    return pd.read_csv('data/soybean.csv')

# Função principal
def encode_and_save():
    dados = load_data()
    encoders = {}

    # Aplica o LabelEncoder em todas as colunas categóricas
    for col in dados.columns:
        if dados[col].dtype == 'object':
            label = LabelEncoder()
            dados[col] = label.fit_transform(dados[col])
            encoders[col] = label  # Salva o encoder da coluna

    # Salva os encoders em um arquivo .pkl
    joblib.dump(encoders, 'label_encoders.pkl')
   

if __name__ == "__main__":
    encode_and_save()