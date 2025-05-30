from data.dados import load_data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
# Dataset
dados = load_data()
# Ler arquivos
print(dados.head())
# Contar valores nulos
print(f'Quantidade de Nulos:\n{dados.isnull().sum()}')
# Verificar quantidade de classes
print(f'Descrição:\n{dados.describe()}')
# Verificar quantidade de classes
print(f'Quantidade de Classes: {dados["class"].nunique()}')

doencas = dados.groupby('class').size().reset_index(name='counts')
print(f'Tipos de doenças e a quantidade de ocorrências de cada uma:\n{doencas}')

# definir o estilo visual do gráfico no Seaborn.
sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 6))
sns.countplot(y="class", data=dados, order=dados["class"].value_counts().index, hue="class", color="skyblue")
plt.title("Distribuição das doenças nas plantas")
plt.xlabel("Tipo de Doença")
plt.ylabel("Quantidade")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Aplicar o LabelEncoder na coluna 'class' para converter as classes em números
labelencoder = LabelEncoder()

encoders = {}

for col in dados.columns:
    if dados[col].dtype == 'object':
        label = LabelEncoder()
        dados[col] = label.fit_transform(dados[col])
        encoders[col] = label  # salva o encoder da coluna

# Calculando a correlação entre 'class' e as outras variáveis
corr_class = dados.corrwith(dados['class'])

# Transformando em um DataFrame para fácil visualização
corr_class_df = corr_class.reset_index(name='correlation')
corr_class_df.columns = ['Variable', 'Correlation']

# Plotando o gráfico de dispersão para a correlação
plt.figure(figsize=(10, 6))  # Ajuste o tamanho da figura conforme necessário
sns.scatterplot(x='Variable', y='Correlation', data=corr_class_df, s=100, marker='o', color='blue')

# Adicionando título e rótulos
plt.title("Correlação da 'class' com as outras variáveis")
plt.xlabel('Variáveis')
plt.ylabel('Correlação')

# Rotacionando os rótulos no eixo x para melhorar a leitura
plt.xticks(rotation=90)

# Exibindo o gráfico
plt.tight_layout()
plt.show()


#Verificar quantidade de classes
qtd_class = dados['class'].nunique()
print(f'Quantidade de classes: {qtd_class}')
