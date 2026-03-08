from fastapi.testclient import TestClient
from app.main import app

"""   L'utilisation de 'with TestClient(app) as client' est très importante ici :
    Cela permet de simuler le démarrage complet de l'API (notre lifespan)
    et donc de charger le modèle en mémoire avant de faire les tests ! """


def test_read_root():
    """
        Test 1: Vérifier que la page d'accueil répond bien
    """
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Bienvenue" in response.json()["message"]

def test_predict_missing_features():
    """
        Test 2: Vérifier que l'API gère bien l'erreur si on lui donne trop peu de colonnes
    """
    with TestClient(app) as client:
        #On fimule un client avec seuelement 3 features (le modèle en attend 239)
        donnees_incompletes = {
            "features": {
                "EXT_SOURCE_2": 0.5,
                "DAYS_BIRTH": -15000,
                "AMT_CREDIT": 500000
            }
        }
        response = client.post("/predict", json=donnees_incompletes)
    #On s'attend à ce que le code 400 (Bad Request) soit renvoyé, et non un crash (500)
    assert response.status_code == 400
    assert "Erreur de prédiction" in response.json()["detail"]

def test_predict_bad_format():
    """Test 3 : Vérifier que FastAPI bloque les requêtes mal formatées"""
    with TestClient(app) as client:
        #On envoie un format qui ne respecte pas le dictionnaire "features" attendu
        mauvais_format = {"cle_inconnue": 123} 
        response = client.post("/predict", json=mauvais_format)
        
        # Le code 422 signifie "Unprocessable Entity" (données non conformes au schéma Pydantic)
        assert response.status_code == 422