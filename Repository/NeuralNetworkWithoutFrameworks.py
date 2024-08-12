import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Mean squared error (MSE) loss function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Input and output data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# Random initialization of weights and bias
np.random.seed(42)
input_dim = X.shape[1]
hidden_dim = 3
output_dim = 1
W1 = np.random.randn(input_dim, hidden_dim)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim)  
b2 = np.zeros((1, output_dim))

# Hyperparameters
learning_rate = 0.1
epochs = 10000

# Neural network training
for epoch in range(epochs):
    # Feedforward
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = sigmoid(z2)

    # Backpropagation
    error = mse_loss(Y, y_pred)
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss = {error}")

    # Gradient calculation
    delta2 = (y_pred - Y) * y_pred * (1 - y_pred)
    dW2 = np.dot(a1.T, delta2)
    db2 = np.sum(delta2, axis=0, keepdims=True)
    delta1 = np.dot(delta2, W2.T) * a1 * (1 - a1)
    dW1 = np.dot(X.T, delta1)
    db1 = np.sum(delta1, axis=0, keepdims=True)

    # Update weights and biases
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

# Testing the trained neural network
print("\nTesting the neural network:")
for i in range(len(X)):
    z1 = np.dot(X[i], W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = sigmoid(z2)
    print(f"Input: {X[i]} - Predicted output: {y_pred}")
