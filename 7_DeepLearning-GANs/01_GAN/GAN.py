'''
kombinieren von Generator und Discriminator
sowie trainieren des GANs
'''

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

    def train(self, epochs, batch_size, sample_interval):
        # Load and rescale dataset
        mnist_data = MNIST()
        x_train, _ = mnist_data.get_train_set()
        x_train = x_train / 127.5 - 1.0 # values between -1 and 1       0 => -1  255 => 1
        # Adversarial ground truth
        valid = np.ones((batch_size, 1))  # np array with Ones 
        fake = np.zeros((batch_size, 1))  # np array with Zeros

        # Start training
        for epoch in range(epochs):
            # TRAINSET IMAGES
            #                  Startwert      Anzahl der Bilder in x_train      Anzahl der Indizes
            idx = np.random.randint(0, x_train.shape[0], batch_size) 
            imgs = x_train[idx]
            # GENERATED IMAGES
            # generiert batch size viele Bilder mit z_dimension-Werten wobei die Werte 0 oder 1 sind (schwarz/weiß)
            noise = np.random.normal(0, 1, (batch_size, self.z_dimension))
            gen_imgs = self.generator.predict(noise)
            # TRAIN DISCRIMINATOR
            d_loss_real = self.discriminator.train_on_batch(imgs, valid) # ruft Training für jede Epoche einzeln auf
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # TRAIN GENERATOR
            noise = np.random.normal(0, 1, (batch_size, self.z_dimension))
            g_loss = self.combined.train_on_batch(noise, valid)
            # SAVE PROGRESS
            if (epoch % sample_interval) == 0:
                print("[D loss: ", d_loss[0], "acc: ", round(d_loss[1]*100,2), "] [G loss: ", g_loss, "]")
                self.sample_images(epoch)


    
    # save sample images
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.z_dimension))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("/home/felix/Desktop/DeepLearning/7_DeepLearning-GANs/01_GAN/images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=200000, batch_size=32, sample_interval=1000)
