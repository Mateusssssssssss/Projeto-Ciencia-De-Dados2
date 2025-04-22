FROM python:3.12-slim

# Define diretório de trabalho
WORKDIR /app

# Atualiza pacotes do sistema e instala dependências essenciais
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    build-essential \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia arquivos do projeto
COPY api/ ./api/
COPY best_model/ ./best_model/
COPY requirements.txt .

# Atualiza pip, setuptools e wheel
RUN pip install --upgrade pip setuptools wheel

# Instala as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta da API
EXPOSE 8000

# Comando para iniciar a aplicação
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]