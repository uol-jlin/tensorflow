import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.preprocessing.image import load_img

base_dir = "./data/"
happy_dir = os.path.join(base_dir, "happy/")
sad_dir = os.path.join(base_dir, "sad/")

print("Sample happy image:")
plt.imshow(load_img(f"{os.path.join(happy_dir, os.listdir(happy_dir)[0])}"))
plt.show()

print("\nSample sad image:")
plt.imshow(load_img(f"{os.path.join(sad_dir, os.listdir(sad_dir)[0])}"))
plt.show()

from tensorflow.keras.preprocessing.image import img_to_array

sample_image = load_img(f"{os.path.join(happy_dir, os.listdir(happy_dir)[0])}")
sample_array = img_to_array(sample_image)
print(f"Each image has shape: {sample_array.shape}")
print(f"The maximum pixel value used is: {np.max(sample_array)}")

"""
Each image has shape: (150, 150, 3)
The maximum pixel value used is: 255.0
"""

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if logs.get('accuracy') is not None and logs.get('accuracy') > 0.999:
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def image_generator():
  train_datagen = ImageDataGenerator(rescale=1/255)
  # generate batches of augmented data
    # - directory: relative path to the directory containing the data
    # - target_size: equal to the resolution of each image (excluding the color dimension)
    # - batch_size: number of images the generator yields when asked for a next batch. 
    # - class_mode: How the labels are represented. Should be one of "binary", "categorical" or "sparse".
  train_generator = train_datagen.flow_from_directory(directory='./data',
                                                      target_size=(150,150),
                                                      batch_size=10,
                                                      class_mode='binary')
  return train_generator

gen = image_generator()

"""
Found 80 images belonging to 2 classes.
"""

from tensorflow.keras import optimizers, losses
from tensorflow.keras.optimizers import RMSprop

def train_happy_sad_model(train_generator):
    # instantiate the callback
    callbacks = myCallback()

    # define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'), 
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # compile the model
    # loss function compatible with last layer of network
    model.compile(loss=losses.binary_crossentropy,
                  optimizer=RMSprop(learning_rate=0.001),
                  metrics=['accuracy'])     


    # train the model
    history = model.fit(x=train_generator,
                        epochs=20,
                        callbacks=[callbacks]
                       )
    
    return history

hist = train_happy_sad_model(gen)

"""
Epoch 1/20
8/8 [==============================] - 3s 243ms/step - loss: 1.0433 - accuracy: 0.4500
Epoch 2/20
8/8 [==============================] - 2s 228ms/step - loss: 0.6369 - accuracy: 0.7000
Epoch 3/20
8/8 [==============================] - 2s 228ms/step - loss: 0.4242 - accuracy: 0.8375
Epoch 4/20
8/8 [==============================] - 2s 215ms/step - loss: 0.3691 - accuracy: 0.8750
Epoch 5/20
8/8 [==============================] - 2s 213ms/step - loss: 0.1674 - accuracy: 0.9375
Epoch 6/20
8/8 [==============================] - 2s 201ms/step - loss: 0.2771 - accuracy: 0.8625
Epoch 7/20
8/8 [==============================] - 2s 215ms/step - loss: 0.0713 - accuracy: 0.9875
Epoch 8/20
8/8 [==============================] - 2s 212ms/step - loss: 0.0653 - accuracy: 0.9625
Epoch 9/20
8/8 [==============================] - 2s 216ms/step - loss: 0.0311 - accuracy: 0.9875
Epoch 10/20
8/8 [==============================] - ETA: 0s - loss: 0.0056 - accuracy: 1.0000
Reached 99.9% accuracy so cancelling training!
8/8 [==============================] - 2s 217ms/step - loss: 0.0056 - accuracy: 1.0000
"""
