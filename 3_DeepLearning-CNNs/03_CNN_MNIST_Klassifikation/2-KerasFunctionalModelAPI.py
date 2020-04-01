# Keras Functional API

import os

import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *

# Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Cast to np.float32
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# Reshape the images to a depth dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Dataset variables
train_size = x_train.shape[0]
test_size = x_test.shape[0]
width, height, depth = x_train.shape[1:]
num_features = width * height * depth
num_classes = 10

# Compute the categorical classes_list
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Model params
lr = 0.001
optimizer = Adam(lr=lr)
epochs = 10
batch_size = 256

# Define the CNN with the Keras Model API
# Input
input_img = Input(shape=x_train.shape[1:])

#                   (Layer)                     (input für den Layer)
x = Conv2D(filters=32, kernel_size=3, padding='same')(input_img)
x = Activation("relu")(x)
x = Conv2D(filters=32, kernel_size=3, padding='same')(x)
x = Activation("relu")(x)
x = MaxPool2D()(x)

x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
x = Activation("relu")(x)
x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
x = Activation("relu")(x)
x = MaxPool2D()(x)

x = Flatten()(x)
x = Dense(units=128)(x)
x = Activation("relu")(x)
x = Dense(units=num_classes)(x)

# Output
y_pred = Activation("softmax")(x)

# Build the model
model = Model(inputs=[input_img], outputs=[y_pred])

# Compile and train (fit) the model, afterwards evaluate the model
model.summary()

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])
model.fit(
    x=x_train, 
    y=y_train, 
    epochs=epochs,
    batch_size=batch_size,
    validation_data=[x_test, y_test])

score = model.evaluate(
    x_test, 
    y_test, 
    verbose=0)
print("Score: ", score)
