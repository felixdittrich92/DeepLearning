'''
siehe Bilder in diesem Ordner
F: Forget Gate -> welche vorherigen Informationen werden verworfen
I: Input Gate -> welche neuen Informationen sind wichtig
O: Output Gate -> welche Informationen werden intern im Cell State gespeichert
C: Candidate State -> welche Informationen werden intern dem Cell State (c) hinzugefügt

h: Hidden State -> Ausgabe der LSTM(Long Short Term Memory) in dem aktuellen Zeitschritt

einmal Keras Implementierung und eigene Implementierung
'''

import random

import numpy as np

import tensorflow as tf

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class LSTMInference:
    def __init__(self, lstm_layer, return_sequences=False):
        self.return_sequences = return_sequences
        self.lstm_layer = lstm_layer
        self.W, self.U, self.b = self.lstm_layer.get_weights()
        self.units = self.b.shape[0] // 4
        self.W_i = self.W[:, :self.units]
        self.W_f = self.W[:, self.units: self.units * 2]
        self.W_c = self.W[:, self.units * 2: self.units * 3]
        self.W_o = self.W[:, self.units * 3:]
        self.U_i = self.U[:, :self.units]
        self.U_f = self.U[:, self.units: self.units * 2]
        self.U_c = self.U[:, self.units * 2: self.units * 3]
        self.U_o = self.U[:, self.units * 3:]
        self.b_i = self.b[: self.units]
        self.b_f = self.b[self.units: self.units * 2]
        self.b_c = self.b[self.units * 2: self.units * 3]
        self.b_o = self.b[self.units * 3:]

    # magic method call überschreiben
    def __call__(self, x):
        # output shape (num_timesteps, units)
        if self.return_sequences:
            self.time_steps = x.shape[0]
            self.h = np.zeros((self.time_steps, self.units))
        # output shape (units)
        else:
            self.h = np.zeros((self.units))
        h_t = np.zeros((1, self.units))
        c_t = np.zeros((1, self.units))
        for t, x_t in enumerate(x):
            x_t = x_t.reshape(1, -1) # (2) => (1, 2)
            c_t, h_t = self.forward_step(x_t, c_t, h_t)
            if self.return_sequences:
                self.h[t] = h_t
            else:
                self.h = h_t
        return self.h

    # Berechnung
    def forward_step(self, x_t, c_t, h_t):
        i_t = sigmoid(np.matmul(x_t, self.W_i) + np.matmul(h_t, self.U_i) + self.b_i)
        f_t = sigmoid(np.matmul(x_t, self.W_f) + np.matmul(h_t, self.U_f) + self.b_f)
        c_tilde = tanh(np.matmul(x_t, self.W_c) + np.matmul(h_t, self.U_c) + self.b_c)
        o_t = sigmoid(np.matmul(x_t, self.W_o) + np.matmul(h_t, self.U_o) + self.b_o)
        c_t = f_t * c_t + i_t * c_tilde
        h_t = o_t * tanh(c_t)
        return c_t, h_t
        

# data set shape = (num_samples, num_timesteps, num_features)
# input shape = (num_timesteps, num_features)
# If return_sequences == True:
# output shape = (num_timesteps, units)
# Else:
# output shape = (1, units)
x = np.random.normal(size=(1, 3, 2))
units = 4
return_sequences = True

# num_features = 2
# units = 4
# h_t shape = (4),        (units)       
# W shape   = (2, 4),     (num_features, units)
# U shape   = (4, 4),     (units, units)
# b shape   = (4),        (units)
# 
# matmul(x, W)      (1, 2)*(2,4) => (4)
# matmul(h, U)      (1, 4)*(4,4) => (4)
# intern + b        (4)+(4)   => (4) 

# Keras Implementation
model = Sequential()
model.add(LSTM(units=units, return_sequences=return_sequences, input_shape=x.shape[1:]))
model.compile(loss="mse", optimizer="Adam")
#model.summary()

# Implementation without Keras
rnn = LSTMInference(lstm_layer=model.layers[0], return_sequences=return_sequences)
output_rnn_own = rnn(x[0]) # 10.5 aufrufen der call Methode
print(output_rnn_own)
print("\n\n")
output_rnn_tf = model.predict(x[[0]])
print(output_rnn_tf) # 10.5
assert np.all(np.isclose(output_rnn_own - output_rnn_tf, 0.0, atol=1e-06))