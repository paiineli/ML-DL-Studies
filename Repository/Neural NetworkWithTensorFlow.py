import numpy as np
import tensorflow as tf

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Input and output data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# Model construction
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, input_shape=(2,), activation='sigmoid'),  # Hidden layer with 3 neurons and sigmoid activation function
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron and sigmoid activation function
])

# Model compilation
model.compile(loss='mean_squared_error', optimizer='sgd')  # Setting the loss function and optimizer

# Model training
history = model.fit(X, Y, epochs=100, batch_size=1, verbose=1)  # Training with 100 epochs and batch size 1

# Displaying training metrics
print("\nTraining History:")
print("Final Loss:", history.history['loss'][-1])

# Model evaluation
print("\nTesting the neural network:")
for i in range(len(X)):
    y_pred = model.predict(np.array([X[i]]))[0][0]
    print(f"Input: {X[i]} - Predicted output: {y_pred}")
