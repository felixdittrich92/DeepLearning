'''
Generator - generiert die Daten bzw. das Bild
'''

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

def build_generator(z_dimension, channels):
    model = Sequential()
    
    model.add(Dense(128 * 7 * 7, input_dim=z_dimension))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))

    model.add(UpSampling2D()) # macht Bild "doppelt so groß" inetwa Gegenteil zum Pooling Layer  7*2 7*2 14, 14
    model.add(Conv2D(128, kernel_size=5, strides=1, padding="same", use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(UpSampling2D())  # 14*2 14*2 28, 28 <- Größe des Ausgangsbildes
    model.add(Conv2D(64, kernel_size=5, strides=1, padding="same", use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(channels, kernel_size=5, strides=1, padding="same", use_bias=False))
    model.add(Activation("tanh"))

    model.summary()
    noise = Input(shape=(z_dimension,)) # Tensor
    img = model(noise)
    return Model(inputs=noise, outputs=img) # speichert zum noise-Vektor das resultierende Bild ab , beeinhaltet komplettes Modell