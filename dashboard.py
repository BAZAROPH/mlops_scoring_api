import streamlit as st
import pandas as pd
import json
import warnings
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently.metrics import DatasetCorrelations
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

#Exécution
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

    # --- Comparaison Avant/Après Optimisation (Données du Profiling) ---
    st.subheader("Impact de l'optimisation du code (Pandas vs NumPy)")
    st.write("Comparatif du temps d'exécution basé sur l'audit de performance (100 requêtes) :")
    
    col_opti1, col_opti2 = st.columns([1, 2])
    with col_opti1:
        st.metric("Version initiale (Pandas)", "0.464 sec")
        st.metric("Version optimisée (NumPy)", "0.111 sec", delta="-76% de latence", delta_color="normal")
        
    with col_opti2:
        df_benchmark = pd.DataFrame({
            "Version du Code": ["1. Pandas (Avant)", "2. NumPy (Après)"],
            "Temps total (secondes)": [0.464, 0.111]
        }).set_index("Version du Code")
        st.bar_chart(df_benchmark)

    st.markdown("---")

    #Transformer les logs en DataFrame pour l'analyse
    df_logs = pd.DataFrame(raw_logs)
    df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"]) #Conversion pour l'axe temporel

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

    #Graphiques : Vue IT (Temporelle) et Vue Métier (Distribution)
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.write("**Vue IT : Évolution de la latence (secondes)**")
        st.line_chart(df_logs.set_index("timestamp")["execution_time_sec"])
        
    with col_chart2:
        st.write("**Vue Métier : Répartition des profils de risque**")
        # On arrondit les scores à la première décimale pour créer des tranches (0.1, 0.2...)
        score_distribution = df_logs["prediction_score"].apply(lambda x: round(x, 1)).value_counts().sort_index()
        st.bar_chart(score_distribution)

    st.markdown("---")

    # ---------- Analyse Data Drift ----------
    st.header("Annalyse de la dérive des données (Data Drift)")
    if st.button("Générer le rapport Evidently"):
        with st.spinner("Analyse en cours..."):
            
            colonnes_communes = list(set(ref_data.columns).intersection(set(curr_data.columns)))

            #Filtrer les colonnes constantes (source courante de NaN/division par zéro dans Evidently)
            colonnes_a_analyser = [
                c
                for c in colonnes_communes
                if ref_data[c].nunique(dropna=False) > 1 or curr_data[c].nunique(dropna=False) > 1
            ]
            if len(colonnes_a_analyser) < len(colonnes_communes):
                st.warning(
                    "Certaines colonnes constantes ont été exclues de l'analyse de dérive (évite les divisions par zéro)."
                )

            #Liste des variables catégorielles (Home Credit Default Risk).
            liste_cat = [
                "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
                "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", 
                "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
                "WEEKDAY_APPR_PROCESS_START", "ORGANIZATION_TYPE", "FONDKAPREMONT_MODE",
                "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE"
            ]
            
            cat_features = [col for col in liste_cat if col in colonnes_a_analyser]
            num_features = [col for col in colonnes_a_analyser if col not in cat_features]

            #Définition du schéma de données
            definition = DataDefinition(
                numerical_columns=num_features,
                categorical_columns=cat_features
            )

            #Conversion sécurisée en Datasets Evidently
            ref_dataset = Dataset.from_pandas(ref_data[colonnes_a_analyser], data_definition=definition)
            curr_dataset = Dataset.from_pandas(curr_data[colonnes_a_analyser], data_definition=definition)

            #Paramétrage des tests de Drift
            drift_report = Report(metrics=[
                DatasetCorrelations(), #<----Concept Drift / ''Drift des métriques''
                DataDriftPreset(method='jensenshannon', num_method='wasserstein', cat_method='psi')
            ])
            # ------------------------------------------------------------------

            #Générer le rapport et récupérer le snapshot
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*invalid value encountered in divide.*",
                )
                #Dans la nouvelle API, on passe les objets `Dataset` que l'on a créés
                snapshot = drift_report.run(
                    reference_data=ref_dataset,
                    current_data=curr_dataset
                )

            #Sauvegarde en html temporaire via le snapshot
            snapshot.save_html("drift_report.html")

            #Affichage dans streamlit
            with open("drift_report.html", "r", encoding="utf-8") as f:
                html_content = f.read()
                
            components.html(html_content, height=1200, scrolling=True)