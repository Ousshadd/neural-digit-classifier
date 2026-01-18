
# Neural Digit Classifier

Application de reconnaissance de chiffres manuscrits développée dans le cadre du Master Informatique, Gouvernance et Transformation Digitale.

## Architecture du Projet
Le projet est conçu de manière modulaire pour assurer la transparence du système :

* **app.py** : Interface utilisateur Streamlit.
* **model.py** : Moteur d'inférence (calcul matriciel z = AW + b).
* **preprocess.py** : Pipeline de normalisation des images en 28x28 pixels.
* **train.py** : Script d'apprentissage utilisant l'optimiseur Adam.
* **model_weights.npz** : Poids et biais extraits du modèle.

## Performance
Le système affiche une précision de 97,2% sur le dataset MNIST. L'interface expose la distribution des probabilités via la fonction Softmax.

## Installation
1. Créer l'environnement virtuel :
   python -m venv venv
2. Activer l'environnement (Windows) :
   venv\Scripts\activate
3. Installer les dépendances :
   pip install -r requirements.txt
4. Lancer l'application :
   streamlit run app.py

## Contexte Academique
* **Etablissement** : Universite Mohammed V, Faculte des Sciences de Rabat.
* **Auteur** : HADDOUCHE Oussama.
* **Encadrant** : Hisham Laanaya.
