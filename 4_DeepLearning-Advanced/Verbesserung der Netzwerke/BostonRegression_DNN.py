import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import random

import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *

from BostonRegression_Data import BOSTON

data = BOSTON()
data.data_preprocessing(preprocess_mode="MinMax")
x_train_splitted, x_val, y_train_splitted, y_val = data.get_splitted_train_validation_set()
x_train, y_train = data.get_train_set()
x_test, y_test = data.get_test_set()
num_targets = data.num_targets

# Save Path
dir_path = os.path.abspath("../DeepLearning/models/BostonRegression/")
data_model_path = os.path.join(dir_path, "boston_model.h5")
# Log Dir
log_dir = os.path.abspath("../DeepLearning/logs/BostonRegression")

def r_squared(y_true, y_pred):
    numerator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_pred)))
    y_true_mean = tf.math.reduce_mean(y_true)
    denominator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_true_mean)))
    r2 = tf.math.subtract(1.0, tf.math.divide(numerator, denominator))
    r2_clipped = tf.clip_by_value(r2, clip_value_min=0.0, clip_value_max=1.0)
    return r2_clipped

# Define the DNN
def model_fn(optimizer, learning_rate, 
             dense_layer_size1, dense_layer_size2, 
             activation_str, dropout_rate, use_bn):
    # Input
    input_house = Input(shape=x_train.shape[1:])
    # Dense Layer 1
    x = Dense(units=dense_layer_size1)(input_house)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    # Dense Layer 2
    x = Dense(units=dense_layer_size2)(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    # Output Layer
    x = Dense(units=num_targets)(x)
    y_pred = Activation("linear")(x)

    # Build the model
    model = Model(inputs=[input_house], outputs=[y_pred])
    opt = optimizer(learning_rate=learning_rate)
    model.compile(
        loss="mse",
        optimizer=opt,
        metrics=[r_squared])
    model.summary()
    return model

# Global params
epochs = 2000
batch_size = 256

params = {
    "optimizer": Adam,
    "learning_rate": 0.001,
    "dense_layer_size1": 128,
    "dense_layer_size2": 64,
    # relu, elu, LeakyReLU
    "activation_str": "relu",
    # 0.05, 0.1, 0.2
    "dropout_rate": 0.00,
    # True, False
    "use_bn": True,
}

rand_model = model_fn(**params)

def schedule_fn2(epoch):
    threshold = 500
    if epoch < threshold:
        return 1e-3
    else:
        return 1e-3 * np.exp(0.005 * (threshold - epoch))

lrs_callback = LearningRateScheduler(
    schedule=schedule_fn2,
    verbose=1)

es_callback = EarlyStopping(
    monitor='val_accuracy',
    patience=200,
    verbose=1,
    restore_best_weights=True)

class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': self.model.optimizer.lr})
        super().on_epoch_end(epoch, logs)

model_log_dir = os.path.join(log_dir, "modelBostonFinal6")
tb_callback = LRTensorBoard(log_dir=model_log_dir)

rand_model.fit(
    x=x_train, 
    y=y_train, 
    verbose=1, 
    batch_size=batch_size, 
    epochs=epochs, 
    callbacks=[tb_callback, lrs_callback, es_callback],
    validation_data=(x_test, y_test))

score = rand_model.evaluate(
    x_test, 
    y_test, 
    verbose=0, 
    batch_size=batch_size)
print("Test performance: ", score)