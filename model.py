import numpy as np

class MLP:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / exp_z.sum(axis=0)

    def forward_propagation(self, X):
        X_input = X.flatten()[:self.input_size].reshape(-1, 1)
        
        Z1 = np.dot(self.W1, X_input) + self.b1
        A1 = self.relu(Z1)
        
        # Calcul dyal l-couche de sortie
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.softmax(Z2)
        
        return A2