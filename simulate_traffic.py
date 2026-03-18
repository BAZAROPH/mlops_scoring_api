import pandas as pd
import requests
import time

print("Chargement des données de référence...")
#charger juste 50 lignes au hasard pour aller vite
df = pd.read_csv("./data/reference_data_test.csv").sample(50)
print(f"{df.shape[0]} lignes chargées. Voici un aperçu :")
# --- CRÉATION DU FAUX DATA DRIFT ---
#On va truquer une colonne importante (ex: on divise l'âge par 2)
#pour que l'outil Evidently panique et détecte une anomalie !
if "DAYS_BIRTH" in df.columns:
    df["DAYS_BIRTH"] = df["DAYS_BIRTH"] * 0.5
    print("Faux Data Drift appliqué sur DAYS_BIRTH !")

#L'URL de l' API en local
API_URL = "http://127.0.0.1:8000/predict"

print("Début de l'envoi des requêtes à l'API...")
reussites = 0

for index, row in df.iterrows():
    #On transforme la ligne Excel en dictionnaire Python
    features_dict = row.to_dict()
    
    #Préparer le format attendu par ton API {"features": {...}}
    payload = {"features": features_dict}
    
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            reussites += 1
            print(f"✅ Requête envoyée avec succès (Client {index})")
        else:
            print(f"Erreur {response.status_code} : {response.text}")
    except requests.exceptions.ConnectionError:
        print("IMPOSSIBLE DE JOINDRE L'API. Est-elle bien lancée dans un autre terminal ?")
        break
        
    #faire une Petite pause pour ne pas saturer l'ordinateur
    time.sleep(0.1)

print(f"Simulation terminée : {reussites} requêtes enregistrées dans les logs !")