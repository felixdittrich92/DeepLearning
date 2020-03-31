# One-Hot-Encoding:
#╔════════════╦═════════════════╦════════╗ 
#║ CompanyName Categoricalvalue ║ Price  ║
#╠════════════╬═════════════════╣════════║ 
#║ VW         ╬      1          ║ 20000  ║
#║ Acura      ╬      2          ║ 10011  ║
#║ Honda      ╬      3          ║ 50000  ║
#║ Honda      ╬      3          ║ 10000  ║
#╚════════════╩═════════════════╩════════╝

#╔════╦══════╦══════╦════════╦
#║ VW ║ Acura║ Honda║ Price  ║
#╠════╬══════╬══════╬════════╬
#║ 1  ╬ 0    ╬ 0    ║ 20000  ║
#║ 0  ╬ 1    ╬ 0    ║ 10011  ║
#║ 0  ╬ 0    ╬ 1    ║ 50000  ║
#║ 0  ╬ 0    ╬ 1    ║ 10000  ║
#╚════╩══════╩══════╩════════╝

# Softmax:
# Aktivierungsfunktion wenn mehr als 2 Klassen vorhergesagt werden sollen
# wandelt Output Vektor in Wahrscheinlichkeiten um und normiert die Werte also ergeben zusammen 1
# Bsp.:  [1.2]               [0.46] zu 46% Klasse 0
#        [0.9] -> Softmax -> [0.34] zu 34% Klasse 1
#        [0.4]               [0.20] zu 20% Klasse 2

# Cross Entropy:
# Fehlerfunktion (loss function) -> anpassen der Gewichte -> start nächste Epoche

import matplotlib.pyplot as plt
import numpy as np

def generate_data_or():
    x = [[0, 0], [1, 1], [1, 0], [0, 1]]
    y = [0, 1, 1, 1]
    return x, y


# y[0] = 0
# OneHot: y[0] = [1, 0] entspricht Klasse 0
# y[1] = 1
# OneHot: y[1] = [0, 1] entspricht Klasse 1
# siehe Ausgabe
def to_one_hot(y, num_classes):
    y_one_hot = np.zeros(shape=(len(y), num_classes)) # 4x2
    for i, y_i in enumerate(y):
        y_oh = np.zeros(shape=num_classes)
        y_oh[y_i] = 1
        y_one_hot[i] = y_oh
    return y_one_hot

x, y = generate_data_or()
y = to_one_hot(y, num_classes=2)
print("One Hot Encoding: \n", y)

p1 = [0.223, 0.613]
p2 = [-0.75, 0.5]
p3 = [0.01, 0.2]
p4 = [0.564, 0.234]
y_pred = np.array([p1, p2, p3, p4])

def softmax(y_pred):
    y_softmax = np.zeros(shape=y_pred.shape)
    for i in range(len(y_pred)):
        exps = np.exp(y_pred[i])
        y_softmax[i] = exps / np.sum(exps)
    return y_softmax

print("before Softmax: \n", y_pred)
y_pred = softmax(y_pred)
print("after Softmax: \n", y_pred)

def cross_entropy(y_true, y_pred):
    num_samples = y_pred.shape[0]
    num_classes = y_pred.shape[1]
    loss = 0.0
    for y_t, y_p in zip(y_true, y_pred):
        for c in range(num_classes):
            loss -= y_t[c] * np.log(y_p[c])
    return loss / num_samples

loss = cross_entropy(y, y_pred)
print("Cross Entropy: \n",loss)