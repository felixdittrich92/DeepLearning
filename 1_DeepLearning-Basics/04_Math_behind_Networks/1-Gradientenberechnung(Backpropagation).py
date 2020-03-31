'''
siehe Gradientenberechnung(Backpropagation).png

Forward:

X - Input
W - Weights
b - Bias
U - next Weights
c - next Bias
y - result (bzw. eigentliches Ergebnis)

Knoten vom + zum - ist die Prediction
y -> - -> hoch 2 ist MSE (Lossfunktion)
berechnet den "Verlust" zwischen der Prediction und dem eigentlichen y (Ergebniswert)

1. W * x + b = signal
2. z = relu(signal)
3. predicted y = U * z + c

Gewichte optimieren:

Backpropagation - Berechnung wird durch den Optimizer angegeben
(Adam, SGD, ...)
Dabei wird vom Ende des Graphen bis zu den Gewichten (rückwärts) partiel abgeleitet

'''

import os

import numpy as np
np.random.seed(3)

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

# Save Path
log_dir = os.path.abspath("../DeepLearning/logs/computation")

# [0, 0] [1, 1] ... [99, 99]
x = np.array([[i, i] for i in range(100)], dtype=np.float32)
# [0] [1] ... [99]
y = np.array([i for i in range(100)], dtype=np.float32).reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

print(x.shape)
print(y.shape)

# Define the DNN
model = Sequential()
model.add(Dense(1, input_shape=(2,), name="hidden"))
model.add(Activation("relu", name="relu"))
model.add(Dense(1, name="output"))
model.summary()

# Train the DNN
lr = 1e-2
optimizer = Adam(lr=lr)

model.compile(
    loss="mse",
    optimizer=optimizer,
    metrics=["mse"])

tb = TensorBoard(
    log_dir=log_dir, 
    embeddings_freq=0,
    write_graph=True)

model.fit(
    x=x_train,
    y=y_train,
    verbose=1,
    batch_size=1,
    epochs=0,
    validation_data=[x_test, y_test],
    callbacks=[tb])

# feste Weights (W) setzen und 0.100 als Bias (b)
model.layers[0].set_weights([np.array([[-0.250], [1.000]]), np.array([0.100])]) 
# feste Weights (U) setzen und 0.125 als Bias (c)
model.layers[2].set_weights([np.array([[1.250]]), np.array([0.125])]) 

##################
loss_object = tf.keras.losses.MeanSquaredError()

def get_gradients(x_test, y_test, model):
    with tf.GradientTape() as tape:
        y_pred = model(x_test, training=True)
        loss_value = loss_object(y_test, y_pred)
    grads = tape.gradient(loss_value, model.trainable_variables)
    return [(grad, weight) for (grad, weight) in zip(grads, model.trainable_variables)]

x_test = np.array([[2, 2]])
y_test = np.array([[2]])

y_pred = model.predict(x_test)
print("Pred: ", y_pred)

# kernel - Gewichtsmatrix
layer_names = ["hidden:kernel", "hidden:bias", "output:kernel", "output:bias"]
gradients = get_gradients(x_test, y_test, model)

for name, (grads, weight) in zip(layer_names, gradients):
    print("Name:\n", name)
    print("Weights:\n", weight.numpy())
    print("Grads:\n", grads.numpy())
    print("\n")