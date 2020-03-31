'''
MaxPooling: minimiert die berechnete Matrix des Conv Layers 
           (nimmt aus entsprechenden Raster nur die höchsten Werte)

pool_size: Größe des Filters Bsp.:(2, 2)

strides: Wert um wieviel der Filter immer verschoben werden soll 
         Stride = (1,1) 1 nach rechts immer und 1 nach unten wenn Zeile zuende

padding: bestimmt Verhalten des Filters an den Rändern des Bildes
        Same - gibt es Output"bild" selbe Größe wie Eingangs"bild" aus (füllt Rest mit auf)
        Valid - sagt Faltung ist ok -> am Ende kleineres Bild als reingegeben wurde

'''

import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image = x_train[0]
image = image.reshape((28, 28))

# Max-Pooling Funktion definieren und auf ein Bild aus dem
# MNIST Dataset anwenden.
def get_kernel_values(i, j, image):
    kernel_values = image[i:i+2, j:j+2]
    return kernel_values

def max_pooling(image):
    # Setup output image as ndarray
    new_rows = image.shape[0]//2
    new_cols = image.shape[1]//2
    output = np.zeros(shape=(new_rows, new_cols), dtype=np.float32)
    # Compute the values
    for i in range(new_rows):
        for j in range(new_cols):
            kernel_values = get_kernel_values(2*i, 2*j, image)
            max_val = np.max(kernel_values.flatten())
            output[i][j] = max_val
    return output

pooling_image = max_pooling(image)

print(image.shape)
print(pooling_image.shape)

# Input und Outputbild des Pooling Layers mit imshow() ausgeben
plt.imshow(image, cmap="gray")
plt.show()

plt.imshow(pooling_image, cmap="gray")
plt.show()