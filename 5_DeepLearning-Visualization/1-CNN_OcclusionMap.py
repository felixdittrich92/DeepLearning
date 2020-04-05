'''
anzeigen der Prediction pro Pixel - wie sicher ist sich das Netz
'''


import os

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

from plotting import *
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

    x = Conv2D(filters=32, kernel_size=3, padding="same")(input_img)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=3, padding="same")(x)
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
log_dir = os.path.abspath("../DeepLearning/logs/occlusion")

class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)

rlop = ReduceLROnPlateau(
    monitor="val_loss", 
    factor=0.9, 
    verbose=1, 
    patience=2)

tb = LRTensorBoard(
    log_dir=log_dir,
    histogram_freq=0)

callbacks = [tb, rlop]

# model.fit(
#     x=x_train, 
#     y=y_train, 
#     verbose=1, 
#     batch_size=batch_size, 
#     epochs=epochs, 
#     validation_data=[x_valid, y_valid], 
#     callbacks=callbacks)

#model.save_weights("../DeepLearning/models/occlusion_cnn.h5")

model.load_weights("../DeepLearning/models/occlusion_cnn.h5")

# Test the CNN
# score = model.evaluate(
#     x=x_test, 
#     y=y_test, 
#     batch_size=batch_size)
# print("Test performance: ", score)

get_occlusion(
    img=x_train[15],
    img_norm=x_test[15],
    label=y_test[15],
    box_size=8,
    model=model)