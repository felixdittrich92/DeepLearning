'''
Autoencoder - Decoding -> Encoding wobei das Bild "runtergebrochen/komprimiert" wird
auf seine wichtigen Features zur Darstellung - ähnlich PCA Dimensionsreduktion
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Fix CuDnn problem
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

from mnistData import *

# Load MNIST dataset
data = MNIST()
x_train, _ = data.get_train_set()
x_test, _ = data.get_test_set()

# Encoded dimension
encoding_dim = 98

# Keras Model: Autoencoder
# Input Tensors
input_img = Input(shape=(28,28,1,))
# Encoder Part
x = Conv2D(8, kernel_size=3, activation="relu", padding="same")(input_img) # 28x28x8
x = MaxPooling2D(padding="same")(x) # 14x14x8
x = Conv2D(4, kernel_size=3, activation="relu", padding="same")(x) # 14x14x4
x = MaxPooling2D(padding="same")(x) # 7x7x4
encoded = Conv2D(2, kernel_size=3, activation="relu", padding="same")(x) # 7x7x2
# Decoder Part
x = Conv2D(4, kernel_size=3, activation="relu", padding="same")(encoded) # 7x7x4
x = UpSampling2D()(x) # 14x14x4
x = Conv2D(8, kernel_size=3, activation="relu", padding="same")(x) # 14x14x8
x = UpSampling2D()(x) # 28x28x8
decoded = Conv2D(1, kernel_size=3, activation="sigmoid", padding="same")(x) # 28x28x1
# Output Tensors
autoencoder = Model(inputs=input_img, outputs=decoded)
 
# Training
autoencoder.compile(optimizer="adadelta", loss="binary_crossentropy")
autoencoder.fit(x_train, x_train, epochs=50, batch_size=128, validation_data=(x_test, x_test))

# Testing
test_images = x_test[:10]
decoded_imgs = autoencoder.predict(test_images)

# PLot test images
plt.figure(figsize=(12,6))
for i in range(10):
    # Original image
    ax = plt.subplot(2 , 10, i+1)
    plt.imshow(test_images[i].reshape(28,28), cmap="gray")
    # Decoded image
    ax = plt.subplot(2 , 10, i+1+10)
    plt.imshow(decoded_imgs[i].reshape(28,28), cmap="gray")
plt.show()