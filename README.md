# Neural Digit Classifier

[cite_start]Application de reconnaissance de chiffres manuscrits de bout en bout, développée dans le cadre du Master Informatique, Gouvernance et Transformation Digitale[cite: 8]. [cite_start]Ce projet vise à transformer un tracé humain en donnée numérique exploitable grâce à l'apprentissage profond[cite: 12, 14, 15].

## Architecture du Projet

[cite_start]Le projet est conçu de manière modulaire pour assurer la transparence et la portabilité du système[cite: 67, 69]:

* [cite_start]**app.py** : Interface utilisateur développée avec Streamlit permettant une interaction en temps réel[cite: 73, 94, 95].
* [cite_start]**model.py** : Moteur d'inférence qui reconstruit le calcul matriciel (z = AW + b) sans utiliser de bibliothèques opaques[cite: 70, 85, 86].
* [cite_start]**preprocess.py** : Pipeline de normalisation des images (binarisation, cadrage et redimensionnement en 28x28 pixels)[cite: 31, 32, 71, 77].
* [cite_start]**train.py** : Script d'apprentissage utilisant l'algorithme d'optimisation Adam sur le dataset MNIST[cite: 58, 72].
* [cite_start]**model_weights.npz** : Fichier contenant les poids et les biais extraits du modèle pour une exécution immédiate[cite: 63, 66, 70].

## Performance et Resultats

[cite_start]Le modèle repose sur un Perceptron Multi-Couches (MLP) entraîné sur 70 000 images[cite: 20, 49].

* [cite_start]**Precision** : Le système affiche une exactitude (Accuracy) de 97,2% sur les données de test[cite: 132].
* [cite_start]**Interpretabilite** : L'interface expose la distribution des probabilités via la fonction Softmax pour quantifier la certitude du modèle[cite: 99, 102, 105].
* [cite_start]**Analyse critique** : Le projet inclut une visualisation des erreurs pour identifier les ambiguïtés entre les classes similaires comme le 4 et le 9[cite: 121, 150, 151].

## Installation et Configuration

L'installation utilise un environnement virtuel (venv) pour isoler les dépendances nécessaires.

### 1. Cloner le projet
git clone https://github.com/Ousshadd/neural-digit-classifier.git
cd neural-digit-classifier

### 2. Creer l'environnement virtuel
python -m venv venv

### 3. Activer l'environnement
**Sur Windows :**
venv\Scripts\activate

**Sur macOS/Linux :**
source venv/bin/activate

### 4. Installer les dependances
pip install -r requirements.txt

### 5. Lancer l'application
streamlit run app.py

## Contexte Academique

* [cite_start]**Etablissement** : Universite Mohammed V, Faculte des Sciences de Rabat[cite: 2, 3].
* [cite_start]**Auteur** : HADDOUCHE Oussama[cite: 5].
* [cite_start]**Encadrant** : Hisham Laanaya[cite: 6].
