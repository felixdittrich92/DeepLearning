import matplotlib.pyplot as plt
import numpy as np

def get_dataset():
    x = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 1, 1, 1])
    return x, y

class Perceptron():
    def __init__(self, epochs, learnrate):
        self.epochs = epochs
        self.learnrate = learnrate
        self.w = list()

    def train(self, x, y):
        N, dim = x.shape
        # Init model
        self.w = np.random.uniform(-1, 1, (dim,1)) # Gleichverteilung [-1, 1]: Weights
        # Gewichte vor dem Training
        print(self.w)
        # Training
        error = 0.0
        for epoch in range(self.epochs):
            choice = np.random.choice(N) # 0 1 2 oder 3 : random Beispiel aus den Daten
            x_i = x[choice] # einzelner Datenpunkt von x
            y_i = y[choice] # einzelner Datenpunkt von y
            y_hat = self.predict(x_i)
            # Check ob Fehler gemacht wurde
            if y_hat != y_i:
                error += 1
                self.update_weights(x_i, y_i, y_hat)
        print("Train Error: ", error / y.shape[0])
        # Gewichte nach dem Training
        print(self.w)

    def test(self, x, y):
        # [0, 0] => 1, [1, 0] = 1
        # y_pred = [1, 1]
        # y = [0, 1]
        y_pred = np.array([self.predict(x_i) for x_i in x])
        # y_p = 1 != y_i = 0 Fehler
        # y_p = 1 == y_i = 1 Korrekt
        # acc = 1 / 2 = 0.5  entspricht accuracy von 50%
        acc = sum(1 for y_p, y_i in zip(y_pred, y) if y_p == y_i) / y.shape[0]
        print("Test Acc: ", acc)

    def update_weights(self, x, y, y_hat):
        for i in range(self.w.shape[0]):
            # delta_wi = learnrate * (y - y_hat) * x[i]
            delta_w_i = self.learnrate * (y - y_hat) * x[i]
            self.w[i] = self.w[i] + delta_w_i

    def activation(self, signal):
        if signal > 0:
            return 1
        else:
            return 0

    def predict(self, x):
        input_signal = np.dot(self.w.T, x) # w = [1, 2] * [3, 4]
        output_signal = self.activation(input_signal)
        return output_signal


x, y = get_dataset()
learnrate = 1.0
epochs = 10

# Perceptron-Object erstellen
p = Perceptron(epochs=epochs, learnrate=learnrate)
p.train(x, y)
p.test(x, y)

