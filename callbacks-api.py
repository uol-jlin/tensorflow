import tensorflow as tf

# Load the Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fmnist.load_data()

# Normalize the images to a 0 to 1 range
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define a callback class to stop training based on condition
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') < 0.4):
            print("\nLoss is lower than 0.4 so cancelling training!")
            self.model.stop_training = True

# Instantiate the callback
callbacks = myCallback()

# Build the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with the custom callback
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
