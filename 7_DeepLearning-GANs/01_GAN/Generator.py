'''
Generator - generiert die Daten bzw. das Bild
'''

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

import numpy as np

def build_generator(z_dimension, img_shape):
    model = Sequential()
    # Hidden Layer 1
    model.add(Dense(256, input_dim=z_dimension))
    model.add(LeakyReLU(alpha=0.2)) # "Knick" bei < 0
    model.add(BatchNormalization(momentum=0.8))
    # Hidden Layer 2
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2)) 
    model.add(BatchNormalization(momentum=0.8))
    # Hidden Layer 3
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2)) 
    model.add(BatchNormalization(momentum=0.8))
    # Output Layer
    model.add(Dense(np.product(img_shape))) # 28 x 28 = 784 Neuronen
    model.add(Activation("tanh")) # auf Intervall -1 bis 1 umformen
    model.add(Reshape(img_shape)) # zurück führen auf 28 x 28 -> erzeugtes Bild
    model.summary()

    noise = Input(shape=(z_dimension,)) # Tensor
    img = model(noise)
    return Model(noise, img) # speichert zum noise-Vektor das resultierende Bild ab , beeinhaltet komplettes Modell