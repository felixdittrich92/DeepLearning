'''
Discriminator - bewertet die vom Generator erzeugten Bilder ob Real oder Fake
'''

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

import numpy as np

def build_discriminator(img_shape):
    model = Sequential()
    # .....
    model.summary()

    img = Input(shape=img_shape)
    d_pred = model(img) # model auf bild aufrufen
    return Model(img, d_pred) # beeinhaltet komplettes Modell