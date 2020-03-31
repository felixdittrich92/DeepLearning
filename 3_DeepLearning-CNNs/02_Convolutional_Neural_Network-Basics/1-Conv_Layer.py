'''
Conv Layer: berechnet mithilfe der Gewichtsmatrizen die Outputmatrix
-> siehe ConvLayer.png

size: Größe des Filters meistens 3x3

depth: Tiefe des Bildes (RGB: 3, Grauwert: 2)

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

np.random.seed(42)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image = x_train[0]
image = image.reshape((28, 28))

kernel = np.random.uniform(low=0.0, high=1.0, size=(2,2))

# Conv Funktion definieren und anschließend plotten
def conv2D(image, kernel):
    rows, cols = image.shape #28x28
    k_size, k_size = kernel.shape #2x2 Filter
    conv_image = np.zeros(shape=(rows, cols), dtype=np.float32) #28x28 padding
    
    for i in range(rows-k_size):
        for j in range(cols-k_size):
            conv_image[i][j] = np.sum(kernel * image[i:i+k_size, j:j+k_size]) # falten des abgedeckten Filterbereichs

    return conv_image

conv_image = conv2D(image, kernel)

# Input und Outputbild des Pooling Layers mit imshow() ausgeben
plt.imshow(image, cmap="gray")
plt.show()

plt.imshow(conv_image, cmap="gray")
plt.show()