from fastapi import FastAPI
from api.routes import router

# Inicializa a aplicação FastAPI e conecta as rotas da API organizadas no router


app = FastAPI()
app.include_router(router)