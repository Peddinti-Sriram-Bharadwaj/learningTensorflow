import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (_, _) = keras.datasets.mnist.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0

# Reshape input data to (28, 28, 1)
x_train = np.reshape(x_train, (-1, 28, 28, 1))

# Define generator model
generator = keras.Sequential([
    keras.layers.Input(shape=(100,)),
    keras.layers.Dense(128, activation=tf.nn.leaky_relu),
    keras.layers.Dense(256, activation=tf.nn.leaky_relu),
    keras.layers.Dense(512, activation=tf.nn.leaky_relu),
    keras.layers.Dense(28*28*1, activation=tf.nn.tanh),
    keras.layers.Reshape((28, 28, 1))
])

# Define discriminator model
discriminator = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation=tf.nn.leaky_relu),
    keras.layers.Dense(256, activation=tf.nn.leaky_relu),
    keras.layers.Dense(128, activation=tf.nn.leaky_relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

# Compile discriminator model
discriminator.compile(loss='binary_crossentropy', optimizer='adam')

# Freeze discriminator weights
discriminator.trainable = False

# Define GAN model
gan = keras.Sequential([
    generator,
    discriminator
])

# Compile GAN model
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Train GAN model
epochs = 100
batch_size = 128
steps_per_epoch = x_train.shape[0] // batch_size

for epoch in range(epochs):
    for step in range(steps_per_epoch):
        # Generate random noise vectors
        noise = np.random.randn(batch_size, 100)

        # Generate fake images using generator
        fake_images = generator.predict(noise)

        # Train discriminator on real and fake images
        real_images = x_train[np.random.choice(x_train.shape[0], batch_size)]
        discriminator_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
        discriminator_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
        discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

        # Freeze discriminator weights
        discriminator.trainable = False

        # Train GAN model
        noise = np.random.randn(batch_size, 100)
        gan_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # Unfreeze discriminator weights
        discriminator.trainable = True

    # Plot generated images
    noise = np.random.randn(25, 100)
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape((25, 28, 28))
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(generated_images[i], cmap='gray')
        plt.axis('off')
    plt.savefig(f'generated_images_epoch_{epoch}.png')
    plt.close()