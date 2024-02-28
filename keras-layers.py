import keras
from keras import layers

layer = layers.Dense(32, activation='relu')
inputs = keras.random.uniform(shape=(10, 20))
outputs = layer(inputs)
Unlike a function, though, layers maintain a state, updated when the layer receives data during training, and stored in layer.weights:

>>> layer.weights
[<KerasVariable shape=(20, 32), dtype=float32, path=dense/kernel>,
 <KerasVariable shape=(32,), dtype=float32, path=dense/bias>]
