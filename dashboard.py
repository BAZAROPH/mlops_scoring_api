import streamlit as st
import pandas as pd
import json
from evidently import Report
from evidently.presets import DataDriftPreset
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="Dashboard MLOps - Scoring Crédit", layout="wide")

st.title("Dashboard de Monitoring et de Data Drift")
st.write("Ce dashboard compare les données d'entrainement (référence) avec les requêtes reçues par l'API (production).")

#1 Charger les données de référence
@st.cache_data
def load_reference_data():
    return pd.read_csv("./data/reference_data.csv")

#2 Charger les données de production depuis nos los API
def log_current_data():
    log_file = "./logs/api_logs.jsonl"
    if not os.path.exists(log_file):
        return pd.DataFrame()  #Retourne un DataFrame vide si le fichier n'existe pas
    
    data = []
    with open(log_file, "r") as f:
        for line in f:
            log_entry = json.loads(line)
            data.append(log_entry["inputs"])
    
    return pd.DataFrame(data)

#Exécuter
try:
    ref_data = load_reference_data()
    st.success(f"Données de référence chargées : {ref_data.shape[0]} lignes")
except Exception as e:
    st.error(f"Erreur lors du chargement des données de référence : {e}")
    st.stop()

curr_data = log_current_data()

if curr_data.empty:
    st.warning("Aucune donnée de production trouvée. Veuillez envoyer des requêtes à l'API pour voir les résultats.")
else:
    st.success(f"Données de production (logs) chargées : {curr_data.shape[0]} requêtes.")

    if st.button("Générer le rapport de Data Drift (Evidently)"):
        with st.spinner("Analyse en cours... Cela peut prendre quelques secondes."):
            #on demande à evidently de calculer le Data Drift
            drift_report = Report(metrics=[DataDriftPreset()])

            #On aligne les cols au cas où
            colonnes_communes = list(set(ref_data.columns).intersection(set(curr_data.columns)))

            #Générer le rapport et récupérer le snapshot
            snapshot = drift_report.run(
                reference_data=ref_data[colonnes_communes],
                current_data=curr_data[colonnes_communes],
            )

            #Sauvegarde en html temporaire via le snapshot
            snapshot.save_html("drift_report.html")

            #Affichage dans streamlit
            with open("drift_report.html", "r", encoding="utf-8") as f:
                html_content = f.read()
                
            components.html(html_content, height=1000, scrolling=True)
