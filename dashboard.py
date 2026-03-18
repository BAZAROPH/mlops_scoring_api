import streamlit as st
import pandas as pd
import json
from evidently import Report
from evidently.presets import DataDriftPreset
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="Dashboard MLOps - Scoring Crédit", layout="wide")

st.title("Dashboard de Monitoring et de Data Drift")
st.write("Ce dashboard surveille la santé de l'API en production (Latence) et la qualité des données (Data Drift).")

#1 Charger les données de référence
@st.cache_data
def load_reference_data():
    return pd.read_csv("./data/reference_data_train.csv")

#2 Charger les logs complet de l'API
def load_logs():
    log_file = "./logs/api_logs.jsonl"
    if not os.path.exists(log_file):
        return [], pd.DataFrame()  #Retourne une liste vide et un DataFrame vide si le fichier n'existe pas
    
    raw_logs = []
    inputs_data = []
    with open(log_file, "r") as f:
        for line in f:
            log_entry = json.loads(line)
            raw_logs.append(log_entry)
            inputs_data.append(log_entry["inputs"])
    return raw_logs, pd.DataFrame(inputs_data)

#Exécuter
try:
    ref_data = load_reference_data()
    st.success(f"Données de référence chargées : {ref_data.shape[0]} lignes")
except Exception as e:
    st.error(f"Fichier `data/reference_data_train.csv` introuvable.")
    st.stop()

raw_logs, curr_data = load_logs()

if curr_data.empty:
    st.warning("Aucune donnée de production trouvée. Veuillez envoyer des requêtes à l'API pour voir les résultats.")
else:
    st.success(f"Données de production (logs) chargées : {len(raw_logs)} requêtes.")
    st.markdown("### Aperçu des données de production (logs) :")

    # ---------- Analyse opérationelle (Latence) ----------
    st.header("Santé opérationnelle de l'API")

    #Transformer les logs en DataFrame pour l'analyse
    df_logs = pd.DataFrame(raw_logs)

    #Calcul des métriques
    avg_latency = df_logs["execution_time_sec"].mean()
    max_latency = df_logs["execution_time_sec"].max()

    #Affichage des gros chiffres
    col1, col2, col3 = st.columns(3)
    col1.metric("Requêtes Traitées", f"{df_logs.shape[0]}")
    col2.metric("Latence Moyenne (s)", f"{avg_latency:.4f} sec")

    #Alerte si la latence dépasse un seuil (ex: 1 sec)
    if max_latency > 1:
        col3.metric("Latence Max (s)", f"{max_latency:.4f} sec", delta="⚠️ Anormale", delta_color="inverse")
    else:
        col3.metric("Latence Max (s)", f"{max_latency:.4f} sec", delta="Normale", delta_color="normal")

    #Graphique de l'évolution du temps de réponse
    st.write("**Évolution du temps de réponse (Latence en secondes)**")
    st.line_chart(df_logs["execution_time_sec"])

    st.markdown("---")

    # ---------- Analyse Data Drift ----------
    st.header("Annalyse de la dérive des données (Data Drift)")
    if st.button("Générer le rapport Evidently"):
        with st.spinner("Analyse en cours..."):
            drift_report = Report(metrics=[DataDriftPreset()])
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