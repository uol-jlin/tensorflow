import tensorflow as tf
fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(f'\nMODEL TRAINING:')
model.fit(training_images, training_labels, epochs=5)
print(f'\nMODEL EVALUATION:')
test_loss = model.evaluate(test_images, test_labels)

"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                 
 max_pooling2d (MaxPooling2  (None, 13, 13, 32)        0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 11, 11, 32)        9248      
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 5, 5, 32)          0         
 g2D)                                                            
                                                                 
 flatten (Flatten)           (None, 800)               0         
                                                                 
 dense (Dense)               (None, 128)               102528    
                                                                 
 dense_1 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 113386 (442.91 KB)
Trainable params: 113386 (442.91 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________

MODEL TRAINING:
Epoch 1/5
1875/1875 [==============================] - 13s 5ms/step - loss: 0.4757 - accuracy: 0.8270
Epoch 2/5
1875/1875 [==============================] - 7s 4ms/step - loss: 0.3204 - accuracy: 0.8830
Epoch 3/5
1875/1875 [==============================] - 8s 4ms/step - loss: 0.2731 - accuracy: 0.8997
Epoch 4/5
1875/1875 [==============================] - 12s 6ms/step - loss: 0.2430 - accuracy: 0.9100
Epoch 5/5
1875/1875 [==============================] - 7s 4ms/step - loss: 0.2181 - accuracy: 0.9184

MODEL EVALUATION:
313/313 [==============================] - 1s 4ms/step - loss: 0.2672 - accuracy: 0.9020
"""
