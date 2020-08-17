'''
Generator - generiert die Daten bzw. das Bild
'''

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

import numpy as np

def build_generator(z_dimension, img_shape):
    model = Sequential()
    # ...
    model.summary()

    noise = Input(shape=(z_dimension,)) # Tensor
    img = model(noise)
    return Model(noise, img) # speichert zum noise-Vektor das resultierende Bild ab , beeinhaltet komplettes Modell