# 🏦 Projet 7 : Déploiement d'un Modèle de Scoring (MLOps)

Ce projet représente la phase finale (Partie 2/2) de la création d'un outil de "Scoring Crédit" pour une société financière. L'objectif est de prendre un modèle de Machine Learning entraîné (Phase 1) et de le **mettre en production** en appliquant les meilleures pratiques **MLOps**.

---

## 🏗️ Explication de l'Architecture Technique

Pour ce projet, nous avons mis en place un pipeline complet allant du code local jusqu'à la surveillance dans le Cloud :

1. **L'API REST (FastAPI) :** C'est le cœur du système. Elle charge le modèle `LightGBM` (format `.joblib`) et expose une route `/predict` qui permet aux applications de demander un score de crédit en temps réel.
2. **La Conteneurisation (Docker) :** Pour s'assurer que l'API fonctionne de la même manière sur n'importe quel ordinateur, elle est enfermée dans un conteneur contenant son propre système Linux (Python 3.11, dépendances système comme `libgomp1`).
3. **Les Tests Automatisés (Pytest) :** Avant tout déploiement, des scripts vérifient que le modèle se charge bien et que l'API répond correctement (taux de couverture vérifié avec `pytest-cov`).
4. **Le CI/CD (GitHub Actions) :** À chaque `git push` sur la branche principale, un robot GitHub lance les tests. Si tout est au vert, il envoie automatiquement la nouvelle version sur les serveurs Cloud.
5. **Le Déploiement Cloud (Hugging Face Spaces) :** L'API est hébergée publiquement et gratuitement, prête à être utilisée par le monde entier.
6. **Le Monitoring & Data Drift (Streamlit + Evidently AI) :** L'API enregistre toutes les requêtes reçues dans un dossier `logs/`. Un tableau de bord interactif compare ces données "en direct" avec les données d'entraînement pour détecter toute dérive (Data Drift).

---

## 📂 Structure du Projet

```text
📁 projet_scoring_mlops/
├── 📁 .github/workflows/ # Pipeline CI/CD automatisé
│   └── ci.yml            # Tests et déploiement sur Hugging Face
├── 📁 app/               # Code source principal
│   ├── __init__.py
│   └── main.py           # API FastAPI et système de logging
├── 📁 data/              # Échantillon de données (référence pour Evidently)
├── 📁 model/             # Le modèle ML sérialisé (.joblib)
├── 📁 tests/             # Scripts de tests automatisés
│   └── test_api.py
├── 📄 dashboard.py       # Interface Streamlit pour le monitoring
├── 📄 simulate_traffic.py# Script générant de fausses requêtes pour tester le drift
├── 📄 Dockerfile         # Recette de création de l'environnement isolé
├── 📄 requirements.txt   # Liste des librairies Python nécessaires
└── 📄 README.md          # La documentation que vous lisez
```

---

## 🚀 Installation & Prérequis

Clonez le dépôt et installez l'environnement de travail en local :

```bash
# 1. Cloner le projet
git clone https://github.com/bazaroph/mlops_scoring_api.git
cd mlops_scoring_api

# 2. Créer un environnement virtuel
python -m venv .venv

# 3. Activer l'environnement (Mac/Linux)
source .venv/bin/activate
# (Sur Windows : .venv\Scripts\activate)

# 4. Installer les librairies
pip install -r requirements.txt
```

---

## 💻 Guide d'Utilisation (Runbook)

### 1. Lancer l'API en local (Développement)

Démarrez le serveur FastAPI avec la commande suivante :

```bash
uvicorn app.main:app --reload
```

👉 L'API tourne sur `http://127.0.0.1:8000`
👉 L'interface de test (Swagger) est sur `http://127.0.0.1:8000/docs`

### 2. Tester le code (Pytest)

Pour vérifier que l'API n'est pas "cassée" suite à une modification de code :

```bash
PYTHONPATH=. pytest tests/ -v --cov=app
```

### 3. Lancer l'API via Docker (Production locale)

Testez la construction du conteneur avant de l'envoyer sur le Cloud (Note : le port interne est 7860 pour matcher avec les exigences de Hugging Face) :

```bash
docker build -t api-scoring .
docker run -p 8000:7860 api-scoring
```

### 4. Surveiller le Data Drift (Monitoring)

Pour tester le tableau de bord de surveillance, vous avez besoin de 3 terminaux ouverts en même temps :

**Terminal 1 :** Lancez l'API pour qu'elle puisse réceptionner les requêtes.
```bash
uvicorn app.main:app --reload
```

**Terminal 2 :** Lancez l'interface web de surveillance.
```bash
streamlit run dashboard.py
```

**Terminal 3 :** Lancez l'attaque simulée. Ce script envoie 50 requêtes à l'API en modifiant volontairement certaines données (ex: l'âge) pour déclencher une alarme de Data Drift.
```bash
python simulate_traffic.py
```

Une fois le script terminé, retournez sur la page web ouverte par Streamlit et générez le rapport Evidently pour observer les résultats de la dérive.

---

<p align="center">
  <i>Projet réalisé par <b>bazaroph</b> pour <b>OpenClassrooms</b>.</i>
</p>
