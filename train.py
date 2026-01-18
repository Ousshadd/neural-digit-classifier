from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import joblib
import numpy as np

# 1. Charger les données
print("Chargement de MNIST complet...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0

# 2. Modèle MLP robuste
clf = MLPClassifier(
    hidden_layer_sizes=(128, 64), 
    max_iter=30, 
    alpha=1e-4,
    solver='adam', 
    random_state=42,
    verbose=True
)

print("Entraînement...")
clf.fit(X, y)

# 3. Sauvegardes
joblib.dump(clf, 'model_mnist.joblib')

# Sauvegarde des poids en .npz pour le livrable
weights = {f'W{i}': w for i, w in enumerate(clf.coefs_)}
biases = {f'b{i}': b for i, b in enumerate(clf.intercepts_)}
np.savez("model_weights.npz", **weights, **biases)

print("Terminé !")