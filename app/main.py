from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import os

#Variable globale pour stocker notre dictionnaire contenant le modèle
ml_models = {}

#1 Gestion de la durée de vie de l'API
@asynccontextmanager
async def lifespan(app: FastAPI):
    #Code qui s'exeécute uniquement au démarrage de l'API
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "model", "model_scoring.joblib")

    try:
        #On charge le modèle et on le stocke dans notre dictionnaire
        ml_models["model_scoring"] = joblib.load(model_path)
        print("Modèle chargé avec succès !")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
    
    yield #C'est ici que l'api tourne et attend les requêtes

    #Code qui s'exécute uniquement à l'arrêt de l'API
    #Vider la mémoire à la fermeture de l'API
    ml_models.clear()
    print("API arrêtée, mémoire libérée.")


#2 Initialisation de l'Api avec le gestionnaire de durée de vie lifespan
app = FastAPI(
    title="API de Scoring Crédit",
    description="Cette API évalue la probabilité de défaut de paiement d'un client.",
    version="1.0.0",
    lifespan=lifespan
)

#3 Définition du modèle de données d'entrée
class ClientData(BaseModel):
    features: dict

#4 rOUTE D'accueil
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de Scoring Crédit. Visitez /docs pour la documentation."}

#5 Route de prédiction
@app.post("/predict")
def predict(data: ClientData):
    #On vérifie si le modèle a bien été chargé au démarrage
    if "model_scoring" not in ml_models:
        raise HTTPException(status_code=500, detail="Le modèle n'est pas prêt.")
    
    try:
        df = pd.DataFrame([data.features])
        model = ml_models["model_scoring"]
        
        #Prédiction (classe 1 = défaut)
        #model.predict_proba renvoie [[proba_classe_0, proba_classe_1]]
        proba_defaut = model.predict_proba(df)[0][1]
        
        return {
            "probabilite_defaut": float(proba_defaut),
            "decision_accorde": bool(proba_defaut < 0.5)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction : {str(e)}")