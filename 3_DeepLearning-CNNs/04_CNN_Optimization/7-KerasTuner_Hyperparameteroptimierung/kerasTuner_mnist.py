# pip3 install keras-tuner
# https://analyticsindiamag.com/how-to-use-keras-tuner-for-hyper-parameter-tuning-of-deep-learning-models/

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import * 
from kerastuner.tuners import RandomSearch
import kerastuner as kt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train=X_train.reshape(len(X_train),28,28,1)
X_test=X_test.reshape(len(X_test),28,28,1)

def create_model(hyperparam):
    model = Sequential()
    model.add(Conv2D(filters=hyperparam.Int('convolution_1',min_value=32, max_value=128, step=16), kernel_size=hyperparam.Choice('convolution_1', values = [3,6]),activation='relu',input_shape=(28,28,1))
    model.add(Conv2D(filters=hyperparam.Int('convolution_2', min_value=64, max_value=128, step=16), kernel_size=hyperparam.Choice('convolution_2', values = [3,6]),activation='relu')
    model.add(Conv2D(filters=hyperparam.Int('convolution_3', min_value=64, max_value=128, step=16), kernel_size=hyperparam.Choice('convolution_3', values = [3,6]),activation='relu')
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation("softmax"))
    model.compile(optimizer=Adam(hyperparam.Choice('learning_rate', values=[1e-2, 1e-3])),loss='binary_crossentropy',metrics=['accuracy'])
return model

tuner_search=RandomSearch(model, objective='val_accuracy', max_trials=5, directory='output', project_name="mnist")
tuner_search.search(X_train,y_train,epochs=5,validation_data=(X_test,y_test))

model=tuner_search.get_best_models(num_models=1)[0]
model.fit(X_train,y_train, epochs=10, validation_data=(X_test,y_test))