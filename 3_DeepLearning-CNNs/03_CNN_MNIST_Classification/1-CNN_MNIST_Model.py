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
print(x_train)
x_test = np.expand_dims(x_test, axis=-1)
print(x_test)

# Dataset variables
train_size = x_train.shape[0]
test_size = x_test.shape[0]
width, height, depth = x_train.shape[1:] # 28, 28, 1
num_features = width * height * depth # 28x28x1
num_classes = 10

# Compute the categorical classes_list
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Model params
lr = 0.001
optimizer = Adam(lr=lr)
epochs = 10
batch_size = 256

# Define the CNN
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=3, padding='same', input_shape=x_train.shape[1:]))
model.add(Activation("relu"))
model.add(Conv2D(filters=32, kernel_size=3, padding='same'))
model.add(Activation("relu"))
model.add(MaxPool2D())

model.add(Conv2D(filters=64, kernel_size=5, padding='same'))
model.add(Activation("relu"))
model.add(Conv2D(filters=64, kernel_size=5, padding='same'))
model.add(Activation("relu"))
model.add(MaxPool2D())

model.add(Flatten())

model.add(Dense(units=128))
model.add(Activation("relu"))

model.add(Dense(units=num_classes))
model.add(Activation("softmax")) # bei Klassifikation über 2 Klassen

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
