# FILEPATH: handwriting_recognition.py

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Bidirectional

from tensorflow.keras import Sequential

# Define the model architecture
time_steps = 28
input_dim = 28
num_classes = 10
# BEGIN: 7f2d5d8d7j3d
time_steps = 28
input_dim = 28
# END: 7f2d5d8d7j3d

model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(time_steps, input_dim)),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=128)
import matplotlib.pyplot as plt
import numpy as np

# Make predictions on the test data
predictions = model.predict(x_test)

# Plot some of the predictions
fig, axs = plt.subplots(2, 5, figsize=(12, 6))
axs = axs.flatten()
for i in range(10):
    img = x_test[i]
    true_label = np.argmax(y_test[i])
    predicted_label = np.argmax(predictions[i])
    axs[i].imshow(img, cmap='gray')
    axs[i].set_title(f'True: {true_label}, Predicted: {predicted_label}')
plt.show()
