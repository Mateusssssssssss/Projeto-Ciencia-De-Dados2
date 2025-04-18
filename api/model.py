# Classe que define o formato dos dados de entrada da API.
# Usada para validação automática com Pydantic e documentação no Swagger.
# Garante que o modelo receba os dados no tipo e formato corretos:
from pydantic import BaseModel
import joblib

encoders = joblib.load('label_encoders.pkl')

class InputData(BaseModel):
    date: str
    plant_stand: str
    precip: str
    temp: str
    hail: str
    crop_hist: str
    area_damaged: str
    severity: str
    seed_tmt: str
    germination: str
    plant_growth: str
    leaves: str
    leafspots_halo: str
    leafspots_marg: str
    leafspot_size: str
    leaf_shread: str
    leaf_malf: str
    leaf_mild: str
    stem: str
    lodging: str
    stem_cankers: str
    canker_lesion: str
    fruiting_bodies: str
    external_decay: str
    mycelium: str
    int_discolor: str
    sclerotia: str
    fruit_pods: str
    fruit_spots: str
    seed: str
    mold_growth: str
    seed_discolor: str
    seed_size: str
    shriveling: str
    roots: str
