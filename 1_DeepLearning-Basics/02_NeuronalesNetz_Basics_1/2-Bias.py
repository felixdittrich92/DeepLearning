# jedes Neuron bekommt Biasebene ausser Input, damit Inputdaten nicht verfälscht werden
# wird verwendet, damit kein 0 rauskommt
# es verschiebt die Aktivierungsfunktion nach links bzw. rechts

import matplotlib.pyplot as plt
import numpy as np

# Bias neuron
b = 1
w_0 = 2.0
shift = b * w_0

w_02 = -2.0
shift2 = b * w_02

# plot function
# f(a) = 0, if x < 0 else 1
data = [0 for a in range(-10, -2)]
data.extend([1 for a in range(-2, 10)])
data_n = [0 for a in range(-10, 0)]
data_n.extend([1 for a in range(0, 10)])
data2 = [0 for a in range(-10, 2)]
data2.extend([1 for a in range(2, 10)])

plt.step(range(-10, 10), data, color='blue')
plt.step(range(-10, 10), data_n, color='red')
plt.step(range(-10, 10), data2, color='black')
plt.xlabel('a')
plt.ylabel('plot(a)')
plt.xlim(-12, 12)
plt.ylim(-0.5, 1.5)
plt.legend(['Verschoben +2', 'Normal', 'Verschoben -2'])

#plt.savefig("step2.png")
plt.show()

# Tanh
# f(a) = tanh(a) = 2 / (1+e^(-2x)) - 1
data = [2 / (1 + np.exp(-2 * (a + shift) )) - 1 for a in range(-10, 10, 1)]
data_n = [2 / (1 + np.exp(-2 * a )) - 1 for a in range(-10, 10, 1)]
data2 = [2 / (1 + np.exp(-2 * (a + shift2) )) - 1 for a in range(-10, 10, 1)]

plt.plot(range(-10, 10), data, color='blue')
plt.plot(range(-10, 10), data_n, color='red')
plt.plot(range(-10, 10), data2, color='black')
plt.xlabel('a')
plt.ylabel('tanh(a)')
plt.xlim(-12, 12)
plt.ylim(-1.0, 2.0)
plt.legend(['Verschoben -2', 'Normal', 'Verschoben +2'])

#plt.savefig("tanh2.png")
plt.show()

# SIGMOID
# sigma(a) = 1 / (1 + e^-a)
data = [1 / (1 + np.exp(-a + shift)) for a in range(-10, 10, 1)]
data_n = [1 / (1 + np.exp(-a)) for a in range(-10, 10, 1)]
data2 = [1 / (1 + np.exp(-a + shift2)) for a in range(-10, 10, 1)]

plt.plot(range(-10, 10), data, color='blue')
plt.plot(range(-10, 10), data_n, color='red')
plt.plot(range(-10, 10), data2, color='black')
plt.xlabel('a')
plt.ylabel('sigmoid(a)')
plt.xlim(-12, 12)
plt.ylim(0.0, 2.0)
plt.legend(['Verschoben -2', 'Normal', 'Verschoben +2'])

#plt.savefig("sigmoid2.png")
plt.show()

# RELU = Rectified Linear Unit
# f(a) = max (0, a)

data = [max(0, a + shift) for a in range(-10, 10, 1)]
data_n = [max(0, a) for a in range(-10, 10, 1)]
data2 = [max(0, a + shift2) for a in range(-10, 10, 1)]

plt.plot(range(-10, 10), data, color='blue')
plt.plot(range(-10, 10), data_n, color='red')
plt.plot(range(-10, 10), data2, color='black')
plt.xlabel('a')
plt.ylabel('relu(a)')
plt.xlim(-12, 12)
plt.ylim(0.0, 5.0)
plt.legend(['Verschoben +2', 'Normal', 'Verschoben -2'])

#plt.savefig("relu2.png")
plt.show()