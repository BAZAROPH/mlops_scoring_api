import cProfile
import pstats
import joblib
import pandas as pd
import numpy as np
import warnings

# Supprimer les warnings de scikit-learn
warnings.filterwarnings('ignore', category=UserWarning, message='.*feature names.*')
warnings.filterwarnings('ignore', category=Warning, message='.*InconsistentVersionWarning.*')

print("Chargement du modèle et des données pour le test...")
model = joblib.load("../model/model_scoring.joblib")
df_test = pd.read_csv("../data/reference_data_train.csv").head(1)
features_dict = df_test.to_dict(orient="records")[0]

# Récupérer les noms de features du modèle entraîné
feature_names = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else list(features_dict.keys())

def prediction_lente_actuelle():
    #Ce que fait l' API actuellement (avec DataFrame Pandas)
    df = pd.DataFrame([features_dict])
    return model.predict_proba(df)[0][1]

def prediction_optimisee():
    #Ce qu'on va mettre en place (NumPy pur, sans DataFrame)
    values = np.array([features_dict[col] for col in feature_names]).reshape(1, -1)
    return model.predict_proba(values)[0][1]

print("PROFILING : VERSION LENTE (Pandas)")
#Lancement de la prédiction 100 fois pour bien voir le temps passé
cProfile.run('for _ in range(100): prediction_lente_actuelle()', 'stats_lente.prof')
p_lente = pstats.Stats('stats_lente.prof')
p_lente.strip_dirs().sort_stats('time').print_stats(5) # Affiche les 5 opérations les plus lentes

print("\nPROFILING : VERSION OPTIMISÉE (Numpy)")
cProfile.run('for _ in range(100): prediction_optimisee()', 'stats_opti.prof')
p_opti = pstats.Stats('stats_opti.prof')
p_opti.strip_dirs().sort_stats('time').print_stats(5)