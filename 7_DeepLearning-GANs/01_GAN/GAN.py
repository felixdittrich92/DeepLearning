from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

import numpy as np
import matplotlib.pyplot as plt

from Generator import *
from Discriminator import *
from Dataset_class import MNIST

# GAN Model class
class GAN():
    def __init__(self):
        # Model parameters
        self.img_rows = 28 
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.z_dimension = 100
        lr = 0.0002
        beta_1 = 0.5
        optimizer = Adam(lr=lr, beta_1=beta_1)
        # BUILD DISCRIMINATOR
        self.discriminator = build_discriminator(self.img_shape)
        self.discriminator.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"]
        )
        # BUILD GENERATOR
        self.generator = build_generator(self.z_dimension, self.img_shape)
        z = Input(shape=self.z_dimension,) # generiert 100 zufällige Werte 
        img = self.generator(z) # erzeugt Bild
        self.discriminator.trainable = False # während Generator trainiert wird darf Discriminator nicht trainiert werden
        d_pred = self.discriminator(img) # Real oder Fake
        self.combined = Model(z, d_pred) # packt an den Generator den Discriminator direkt dran um zu überprüfen ist erzeugtes Bild gut(wirkt real) oder schlecht(wirkt fake)
        self.combined.compile(
            loss="binary_crossentropy",
            optimizer=optimizer)

    def train(self):
        pass

    def sample_images(self):
        pass


if __name__ == '__main__':
    gan = GAN()
    gan.train()
