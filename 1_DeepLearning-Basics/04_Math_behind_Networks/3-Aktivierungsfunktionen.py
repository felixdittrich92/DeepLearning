import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def relu(x):
    if x > 0: return x
    else: return 0

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

# y = ReLU(wx + b)
# y = sigma(wx + b)
# shift = -b/w   # Verschiebung bei w > 1
w = 1  # Gewicht
b = -4 # Bias
# shift = 2
act = relu # sigmoid

x = np.linspace(start=-10, stop=10, num=5000)
y_act = np.array([act(xi * w + b) for xi in x])
y = np.array([act(xi * 1 + 0) for xi in x])

plt.figure(figsize=(8,5))
plt.grid(True)
plt.plot(x, y, color="blue")
plt.plot(x, y_act, color="red")
plt.show()