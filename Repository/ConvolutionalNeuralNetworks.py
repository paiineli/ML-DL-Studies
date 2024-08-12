import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the MNIST dataset (included in Keras)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Data preprocessing
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255.0

# Create the Convolutional Neural Network model
model = tf.keras.models.Sequential([
    # Convolutional layer with 32 filters of size 3x3
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Max pooling layer with 2x2 window size
    layers.MaxPooling2D((2, 2)),
    # Convolutional layer with 64 filters of size 3x3
    layers.Conv2D(64, (3, 3), activation='relu'),
    # Max pooling layer with 2x2 window size
    layers.MaxPooling2D((2, 2)),
    # Flatten layer to convert data into a 1D vector
    layers.Flatten(),
    # Fully connected layer with 128 neurons and ReLU activation function
    layers.Dense(128, activation='relu'),
    # Output layer with 10 neurons and softmax activation function
    layers.Dense(10, activation='softmax'),
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
