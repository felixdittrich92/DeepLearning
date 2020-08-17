# Discriminator - bewertet die vom Generator erzeugten Bilder ob Real oder Fake

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

import numpy as np

def build_discriminator(img_shape):
    model = Sequential()
    model.add(Flatten(input_shape=img_shape)) # "ausrollen"
    # Hidden Layer 1
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2)) # "Knick" bei < 0
    # Hidden Layer 2
    model.add(Dense(512))
    odel.add(LeakyReLU(alpha=0.2)) 
    # Output Layer
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.summary()

    img = Input(shape=img_shape)
    d_pred = model(img) # model auf bild aufrufen
    return Model(img, d_pred) # beeinhaltet komplettes Modell