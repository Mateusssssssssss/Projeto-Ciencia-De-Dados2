# APIRouter: separar e agrupar rotas.
from fastapi import APIRouter, HTTPException
from api.model import InputData
from api.predict import prediction
# organizar as rotas da API de forma modular.
router = APIRouter()
#Decorador para criar uma rota POST na API FastAPI.
# A rota é acessada através do caminho "/predict".
@router.post("/predict")
def predict(data: InputData):
    try:
        return prediction(data)
    except Exception as e: # Captura qualquer exceção que ocorra durante a previsão.
        raise HTTPException(status_code=500, detail="Erro interno na previsão")