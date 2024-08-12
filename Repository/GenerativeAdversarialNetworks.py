import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Define the generator
def build_generator():
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(100,)))
    model.add(layers.Dense(784, activation='tanh'))
    model.add(layers.Reshape((28, 28, 1)))
    return model

# Define the discriminator
def build_discriminator():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Function to train the discriminator
def train_discriminator(discriminator, real_images, fake_images):
    real_labels = np.ones((real_images.shape[0], 1))
    fake_labels = np.zeros((real_images.shape[0], 1))

    real_loss = discriminator.train_on_batch(real_images, real_labels)
    fake_loss = discriminator.train_on_batch(fake_images, fake_labels)

    return real_loss, fake_loss

# Function to train the GAN
def train_gan(generator, discriminator, gan, batch_size, latent_dim):
    fake_latent = np.random.randn(batch_size, latent_dim)
    gan_labels = np.ones((batch_size, 1))

    gan_loss = gan.train_on_batch(fake_latent, gan_labels)

    return gan_loss

# Function to visualize generated images
def visualize_generated_images(generator, latent_dim, num_samples=10):
    noise = np.random.randn(num_samples, latent_dim)
    generated_images = generator.predict(noise)

    plt.figure(figsize=(10,10))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow((generated_images[i, :, :, 0] + 1) / 2, cmap='gray')  # Rescale to [0, 1]
        plt.axis('off')
    plt.show()

# Parameters
latent_dim = 100
batch_size = 128
epochs = 10000

# Build and compile models
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer='adam')

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

discriminator.trainable = False
gan_input = tf.keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Load real data (e.g., MNIST)
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

# Train the GAN
for epoch in range(epochs):
    idx = np.random.randint(0, train_images.shape[0], batch_size)
    real_images = train_images[idx]

    fake_latent = np.random.randn(batch_size, latent_dim)
    fake_images = generator.predict(fake_latent)

    d_loss_real, d_loss_fake = train_discriminator(discriminator, real_images, fake_images)
    gan_loss = train_gan(generator, discriminator, gan, batch_size, latent_dim)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, D Loss Real: {d_loss_real}, D Loss Fake: {d_loss_fake}, GAN Loss: {gan_loss}")
        visualize_generated_images(generator, latent_dim)
