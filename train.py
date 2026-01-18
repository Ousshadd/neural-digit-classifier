import numpy as np
from sklearn.datasets import fetch_openml
from model import MLP

def train_model():
    print("Chargement des données MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data / 255.0
    y = mnist.target.astype(int)
    y_one_hot = np.eye(10)[y].T

    # Architecture
    model = MLP(input_size=784, hidden_size=128, output_size=10)
    
    # He Initialization
    model.W1 = np.random.randn(128, 784) * np.sqrt(2./784)
    model.b1 = np.zeros((128, 1))
    model.W2 = np.random.randn(10, 128) * np.sqrt(2./128)
    model.b2 = np.zeros((10, 1))

    # Hyperparamètres m-hassenin
    lr = 0.1
    epochs = 50
    batch_size = 128
    m = X.shape[0]

    print(f"Entraînement en cours (Epochs: {epochs})...")
    for epoch in range(epochs):
        # Mélanger les données (Shuffle)
        permutation = np.random.permutation(m)
        X_shuffled = X[permutation].T
        y_shuffled = y_one_hot[:, permutation]

        for i in range(0, m, batch_size):
            # Mini-batch
            X_batch = X_shuffled[:, i:i+batch_size]
            y_batch = y_shuffled[:, i:i+batch_size]
            
            # Forward
            Z1 = np.dot(model.W1, X_batch) + model.b1
            A1 = model.relu(Z1)
            Z2 = np.dot(model.W2, A1) + model.b2
            A2 = model.softmax(Z2)
            
            # Backprop
            m_batch = X_batch.shape[1]
            dZ2 = A2 - y_batch
            dW2 = (1./m_batch) * np.dot(dZ2, A1.T)
            db2 = (1./m_batch) * np.sum(dZ2, axis=1, keepdims=True)
            dZ1 = np.dot(model.W2.T, dZ2) * (Z1 > 0)
            dW1 = (1./m_batch) * np.dot(dZ1, X_batch.T)
            db1 = (1./m_batch) * np.sum(dZ1, axis=1, keepdims=True)
            
            # Update
            model.W1 -= lr * dW1
            model.b1 -= lr * db1
            model.W2 -= lr * dW2
            model.b2 -= lr * db2
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch} terminée")

    np.savez('model_weights.npz', W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)
    print("Nouveaux poids sauvegardés !")

if __name__ == "__main__":
    train_model()