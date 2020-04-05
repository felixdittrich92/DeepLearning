'''
komplette Visualisierung der Schritte des Netzes mit Keract -> ziemlich cool
'''

import os

import keract

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *

from DogsCats_Dataset_class import DOGSCATS

data = DOGSCATS()
data.data_augmentation(augment_size=15000)
x_train, y_train = data.x_train, data.y_train
x_test, y_test = data.x_test, data.y_test
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

batch_size = 128
epochs = 15
train_size, width, height, depth = x_train.shape
test_size, num_classes = y_test.shape
width, height, depth = x_train.shape[1:]

# Define the CNN
def create_model():
    input_img = Input(shape=(width, height, depth))

    x = Conv2D(filters=32, kernel_size=3, padding="same", name="heatmap1")(input_img)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=3, padding="same", name="heatmap2")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.1)(x)

    x = Conv2D(filters=64, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=64, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)

    x = Conv2D(filters=96, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=96, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.3)(x)

    x = Conv2D(filters=128, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=128, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.3)(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(num_classes, name="features")(x)
    output_pred = Activation("softmax")(x)

    optimizer = Adam()
    model = Model(
        inputs=input_img, 
        outputs=output_pred)
    model.compile(
        loss="categorical_crossentropy", 
        optimizer=optimizer, 
        metrics=["accuracy"])
    model.summary()
    return model

model = create_model()

model.load_weights("../DeepLearning/models/heat_cnn.h5")

grads = keract.get_gradients_of_activations(
        model,
        x_test[[20]],
        y_test[[20]],
        layer_name='heatmap1')

keract.display_heatmaps(
        grads,
        x_test[20]*255.0)

activations = keract.get_activations(
        model,
        x_test[[20]])

keract.display_activations(
        activations)