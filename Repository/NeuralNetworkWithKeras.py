import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input

# Input and output data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# Model construction
model = Sequential()
model.add(Input(shape=(2,)))  # Define explicitly the input shape
model.add(Dense(3, activation='sigmoid'))  # Layer with 3 neurons and sigmoid activation function
model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 neuron and sigmoid activation function

# Model compilation
model.compile(loss='mean_squared_error', optimizer='sgd')  # Setting the loss function and optimizer

# Model training
model.fit(X, Y, epochs=100, batch_size=1, verbose=1)  # Training with 100 epochs and batch size 1

# Model evaluation
print("\nTesting the neural network:")
for i in range(len(X)):
    y_pred = model.predict(np.array([X[i]]))[0][0]
    print(f"Input: {X[i]} - Predicted output: {y_pred}")
