MNIST Streamlit Classifier

Application de reconnaissance de chiffres manuscrits développée pour le Master Informatique, Gouvernance et Transformation Digitale.

Architecture Technique

Le projet est structuré en modules distincts pour garantir la transparence et la portabilité:

    app.py : Interface utilisateur Streamlit pour une prédiction en temps réel.

    model.py : Moteur d'inférence reconstruisant le calcul matriciel z=AW+b.

    preprocess.py : Pipeline de normalisation des images au standard 28x28 (binarisation, cadrage, centrage).

    train.py : Algorithme d'apprentissage utilisant l'optimiseur Adam.

    model_weights.npz : Stockage des poids (W) et biais (b) pour un système auditable sans "boîte noire".

Résultats

    Modèle : Perceptron Multi-Couches (MLP).

    Précision : 97,2% sur le dataset MNIST.

    Interprétabilité : Affichage dynamique de l'indice de confiance et de la distribution des probabilités via Softmax.

Installation et Configuration
1.Cloner le projet :
	git clone git@github.com:Ousshadd/neural-digit-classifier.git
        cd neural-digit-classifier	
2.Créer l'environnement virtuel:
	python -m venv venv
3.Activer l'environnement :
	Windows : venv\Scripts\activate

	macOS/Linux : source venv/bin/activate
4. Installer les dépendances :
	pip install -r requirements.txt
5. Lancer l'application :
	streamlit run app.py
