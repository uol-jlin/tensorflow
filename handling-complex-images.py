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
