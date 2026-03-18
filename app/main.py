from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import os
from datetime import datetime
import time
import json

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
    
    #Lancer le chronomètre pour mesurer le temps d'exécution
    start_time = time.time()
    
    try:
        #---- /df = pd.DataFrame([data.features])
        model = ml_models["model_scoring"]
        # On s'assure de respecter l'ordre exact des colonnes attendu par le modèle
        feature_names = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else list(data.features.keys())

        # On crée le tableau 2D pour scikit-learn/LightGBM
        import numpy as np
        features_array = np.array([data.features.get(col, 0) for col in feature_names]).reshape(1, -1)
        
        
        #Prédiction (classe 1 = défaut)
        
        #---/ model.predict_proba renvoie [[proba_classe_0, proba_classe_1]]
        #---/ proba_defaut = model.predict_proba(df)[0][1]

        proba_defaut = model.predict_proba(features_array)[0][1]
        
        execution_time = time.time() - start_time

        #Enregistrer la prédiction dans un fichier log
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "execution_time_sec": execution_time,
            "prediction_score": float(proba_defaut),
            "inputs": data.features
        }
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base_dir, "..", "logs")

        #On crée le dossier logs s'il n'existe pas
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "api_logs.jsonl")

        #On écrit la prédiction dans le fichier log au format JSONL (une ligne = une prédiction)
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        return {
            "probabilite_defaut": float(proba_defaut),
            "decision_accorde": bool(proba_defaut < 0.5)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction : {str(e)}")
    
    # test cache