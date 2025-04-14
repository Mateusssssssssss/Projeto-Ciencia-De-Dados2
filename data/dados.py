import pandas as pd
def load_data():
    # Ler o CSV
    dados = pd.read_csv('data/soybean.csv')
    return dados

