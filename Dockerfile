#On part d'un linex légé avec python3.10
FROM python:3.11-slim

#On définit le dossier de travail dans le conteneur
WORKDIR /code

#On installe la librairie système pour LightGBM
RUN apt-get update && apt-get install -y libgomp1

#On copie le fichier des dépendances Python
COPY ./requirements.txt /code/requirements.txt

#On installe les libraires python
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#On copie tout le reste dans le conteneur
COPY ./app /code/app
COPY ./model /code/model

#Exposer l'api qui va communiquer, sur le port 7860 (Port exigé par Hugging Face)
EXPOSE 7860

#La commande qui va s'exécuter au démarrage du conteneur
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]