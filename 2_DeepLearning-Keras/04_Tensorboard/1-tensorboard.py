import os

import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *   # für Tensorboard

# Log erstellen/speichern
dir_path = os.path.abspath("../DeepLearning/logs") # Linux und Windows

# Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Cast to np.float32
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# Dataset Variablen
train_size = x_train.shape[0]
test_size = x_test.shape[0]
num_features = 784 # 28x28
num_classes = 10

# kategorisieren
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# input Daten reshapen
x_train = x_train.reshape(train_size, num_features)
x_test = x_test.reshape(test_size, num_features)

# Modell Parameter
init_w = TruncatedNormal(mean=0.0, stddev=0.01)
init_b = Constant(value=0.0)
lr = 0.001
optimizer = Adam(lr=lr)
epochs = 20
batch_size = 256 # [32, 1024]  Werte dazwischen gibt an wieviele Datenpunkte parrallel verwendet werden zum trainieren

# Modell definieren
model = Sequential()

model.add(Dense(units=500, kernel_initializer=init_w, bias_initializer=init_b, input_shape=(num_features, )))
model.add(Activation("relu"))

model.add(Dense(units=300, kernel_initializer=init_w, bias_initializer=init_b))
model.add(Activation("relu"))

model.add(Dense(units=100, kernel_initializer=init_w, bias_initializer=init_b))
model.add(Activation("relu"))

model.add(Dense(units=num_classes, kernel_initializer=init_w, bias_initializer=init_b))
model.add(Activation("softmax"))
model.summary()

# Modell kompilieren, trainieren und evaluieren
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])

# Tensorboard Callback
tb = TensorBoard(
    log_dir=dir_path,
    histogram_freq=1,  # jede Epoche 2 = alle 2 Epochen etc.
    write_graph=True)

model.fit(
    x=x_train, 
    y=y_train, 
    epochs=epochs,
    batch_size=batch_size,
    validation_data=[x_test, y_test],
    callbacks=[tb]) # benötigt für Tensorboard

score = model.evaluate(
    x_test, 
    y_test, 
    verbose=0)
print("Score: ", score)


# USE: in Konsole tensorboard --logdir LOGSORDNER

# mehrere Modelle vergleichen -> für jedes Modell in Logs Unterordner erstellen und mit Tensorboard den Oberordner angeben