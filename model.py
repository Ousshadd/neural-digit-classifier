import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

class SimpleMLP:
    """
    Une implémentation simple pour charger les poids du fichier .npz
    et effectuer des prédictions sans dépendre de scikit-learn dans l'app finale.
    """
    def __init__(self, weights_path="model_weights.npz"):
        try:
            self.data = np.load(weights_path)
            # On compte le nombre de couches (W0, W1, etc.)
            self.num_layers = len([f for f in self.data.files if f.startswith('W')])
        except Exception as e:
            print(f"Erreur lors du chargement des poids : {e}")

    def predict_proba(self, X):
        """ Propagation avant (Forward propagation) """
        a = X
        for i in range(self.num_layers):
            W = self.data[f'W{i}']
            b = self.data[f'b{i}']
            
            z = np.dot(a, W) + b
            
            # Application de l'activation
            if i < self.num_layers - 1:
                a = relu(z)  # Couches cachées
            else:
                a = softmax(z) # Couche de sortie
        return a

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

def get_model():
    """ Fonction utilitaire pour initialiser le modèle """
    return SimpleMLP()