'''
Preprocessing sollte auf Daten immer angewandt werden, da es nur bessere Ergebnisse liefern kann.
wird nur auf die Trainingsdaten angewandt - wenn das Netz auf neue unbekannte Daten
angewandt wird müssen diese davor auch preprocessing durchlaufen, da das Netz die Daten sonst nicht einordnen kann.
'''


import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class MNIST:
    def __init__(self):
        # Load the data set
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        # Convert to float32
        self.x_train = self.x_train.astype(np.float32)
        self.y_train = self.y_train.astype(np.float32)
        self.x_test = self.x_test.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)
        # Reshape the x data to shape (num_examples, width, height, depth)
        self.x_train = np.expand_dims(self.x_train, axis=-1)
        self.x_test = np.expand_dims(self.x_test, axis=-1)
        # Save important data attributes as variables
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.num_classes = 10
        # Reshape the y data to one hot encoding
        self.y_train = to_categorical(self.y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(self.y_test, num_classes=self.num_classes)
        # Preprocess the data
        # MinMaxScaler [0, 255] => [0.0, 1.0]
        # StandardScaler [0, 255] => mean=0, std=1     std-Standardabweichung
        self.scaler = StandardScaler()
        self.scaler.fit(self.x_train.reshape(self.train_size, 784)) 
        # Mittelwert abziehen und durch Standarabweichung teilen
        self.x_train = self.scaler.transform(self.x_train.reshape(self.train_size, 784))
        #print(self.x_train)
        self.x_test = self.scaler.transform(self.x_test.reshape(self.test_size, 784))
        # Reshapen da das Netz die Daten im Vektorformat benötigt
        self.x_train = self.x_train.reshape((self.train_size, self.width, self.height, self.depth))
        #print(self.x_train)
        self.x_test = self.x_test.reshape((self.test_size, self.width, self.height, self.depth))

    def get_train_set(self):
        return self.x_train, self.y_train

    def get_test_set(self):
        return self.x_test, self.y_test

if __name__ == "__main__":
    mnist = MNIST()

    x_train, y_train = mnist.get_train_set()
    print(np.max(x_train)) # 255 weiß
    print(np.min(x_train)) # 0 schwarz
    print(np.mean(x_train)) # Mittelwert
    print(np.std(x_train)) #
