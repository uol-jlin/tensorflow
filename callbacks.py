import os
import tensorflow as tf
from tensorflow import keras

# Get current directory and construct path for MNIST data
current_dir = os.getcwd()
data_path = os.path.join(current_dir, "data/mnist.npz")

# Load the MNIST dataset
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# Normalize the input data
x_train = x_train / 255.0

# Print the shape of the data
data_shape = x_train.shape
print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.99:
            print("\nReached 99% accuracy so cancelling training")
            self.model.stop_training = True

def train_mnist(x_train, y_train):
    callbacks = MyCallback()
    model = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax),
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
    return history

hist = train_mnist(x_train, y_train)
