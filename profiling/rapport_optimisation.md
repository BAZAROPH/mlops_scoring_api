# Rapport d'Optimisation des Performances de l'API (Étape 4)

## 1. Contexte et Objectif
L'API de Scoring Crédit ayant été déployée et monitorée, nous avons procédé à une analyse de ses performances en production. L'objectif est de réduire le temps d'inférence (latence) pour garantir une réponse rapide aux requêtes, tout en s'assurant de ne créer aucune régression sur la précision du modèle.

## 2. Profiling et Identification des Goulots d'Étranglement
Nous avons utilisé la bibliothèque `cProfile` pour chronométrer l'exécution interne de la prédiction. 

**Diagnostic :** Les résultats du profiling ont révélé que la majeure partie du temps d'exécution (overhead) n'était pas due au modèle lui-même, mais à la transformation des données JSON en un objet `pandas.DataFrame` (fichier interne `construction.py`). Pandas est lourd pour des prédictions unitaires en temps réel.

## 3. Stratégie d'Optimisation Implémentée
Conformément aux stratégies recommandées, nous avons procédé à une **optimisation de code** en supprimant la dépendance à Pandas pour la phase d'inférence :
* **Avant (Pandas) :** `df = pd.DataFrame([data.features])`
* **Après (NumPy) :** `features_array = np.array([data.features.get(col, 0) for col in feature_names]).reshape(1, -1)`

Nous avons remplacé l'instanciation du DataFrame par un tableau `NumPy` pur, qui est traité nativement par scikit-learn/LightGBM. L'ordre des colonnes est sécurisé en lisant l'attribut `feature_names_in_` du modèle.

## 4. Résultats et Preuves
Les tests de profilage sur 100 prédictions consécutives ont donné les résultats suivants :
* **Version Pandas :** ~1 574 000 appels de fonctions en **0.464 secondes**.
* **Version NumPy :** ~317 000 appels de fonctions en **0.111 secondes**.

**Conclusion :** Le temps d'inférence a été divisé par plus de 4 (réduction de ~76% de la latence). La précision reste strictement identique.

## 5. Déploiement
Cette version optimisée a été intégrée dans `app/main.py` et automatiquement déployée en production via notre pipeline CI/CD sur Hugging Face Spaces.