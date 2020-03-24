# Gradientenabstiegsverfahren = Backpropagation
# tastet sich anhand einer angegebenen Lernrate bis zum globalen Minimum 
# am Ende jeder Epoche wird die Lernrate angepasst geregelt durch Optimizer
# Optimizer : Adam ist vorzugsweise am schnellsten

import matplotlib.pyplot as plt
import numpy as np
from helper import *

x0 = np.random.uniform(-2, 2)
x1 = np.random.uniform(-2, 2)
y = f(x0, x1)

print("\n\nGlobale Minimum bei: ", 1, 1)
print("Starte bei x = ", x0, x1)
print("Mit f(x) = ", y)
plot_rosenbrock(x0=x0, x1=x1)

eta = 0.005
it = 0
stop_iter = 1000

downhill_points = []

while it < stop_iter:
    x0 = x0 - eta * f_prime_x0(x0, x1)
    x1 = x1 - eta * f_prime_x1(x0, x1)
    it += 1
    fx = f(x0, x1)
    if it % 100 == 0:
        downhill_points.append([x0, x1])

print("Solution: ", fx)
print("x0 = ", x0)
print("x1 = ", x1)
plot_rosenbrock(downhill=downhill_points, x0=x0, x1=x1)