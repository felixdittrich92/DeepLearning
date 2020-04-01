import os

from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *

from Dataset_class import *

mnist = MNIST()
mnist.data_augmentation(augment_size=10000)
mnist.data_preprocessing(preprocess_mode="MinMax")
x_train, y_train = mnist.get_train_set()
x_test, y_test = mnist.get_test_set()
num_classes = mnist.num_classes

# Model params
lr = 0.001
optimizer = Adam(lr=lr)
epochs = 30
batch_size = 128

# Define the CNN
input_img = Input(shape=x_train.shape[1:])

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