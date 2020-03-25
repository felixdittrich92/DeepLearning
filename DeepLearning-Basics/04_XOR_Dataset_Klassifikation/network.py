import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# XOR Dataset
def get_dataset():
    x = np.array([[0,0], [1,0], [0,1], [1,1]]).astype(np.float32)
    y = np.array([0, 1, 1, 1]).astype(np.float32)
    return x, y

x, y = get_dataset()
x_train, y_train = x, y
x_test, y_test = x, y

# Dataset Variablen
features = 2
classes = 2
target = 1

# Model Variablen
hidden_layer_size = 20
nodes = [features, hidden_layer_size, target] # input, hidden, output - layers
train_size = x_train.shape[0]
test_size = x_test.shape[0]
epochs = 500

class Model:
    def __init__(self):
        # weights und biases erstellen
        # weights (Metriken)
        # Input zu Hidden Layer
        self.W1 = tf.Variable(tf.random.truncated_normal(shape=[nodes[0], nodes[1]], stddev=0.1), name="W1") 
        # Hidden zu Output Layer
        self.W2 = tf.Variable(tf.random.truncated_normal(shape=[nodes[1], nodes[2]], stddev=0.1), name="W2")
        # Biases (Vektoren)
        # Bias vom Input Layer zum Hidden Layer
        self.b1 = tf.Variable(tf.constant(0.0, shape=[nodes[1]]), name="b1")
        # Bias vom Hidden Layer zum Output Layer
        self.b2 = tf.Variable(tf.constant(0.0, shape=[nodes[2]]), name="b2")
        self.variables = [self.W1, self.W2, self.b1, self.b2]
        # Model Variablen
        self.learning_rate = 0.001
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.current_loss_val = None

    def get_variables(self):
        return {var.name: var.numpy() for var in self.variables}

    def predict(self, x):
        # Input Layer
        input_layer = x
        # Vom Input zum Hidden Layer
        # h = (x * W1) + b1
        hidden_layer = tf.math.add(tf.linalg.matmul(input_layer, self.W1), self.b1)
        # ReLU = max(val, 0.0) Aktivierungsfunktion
        hidden_layer_act = tf.nn.relu(hidden_layer)
        # Vom Hidden zum Output Layer
        # o = (h * W2) + b2
        output_layer = tf.math.add(tf.linalg.matmul(hidden_layer_act, self.W2), self.b2)
        # sigmoid = 1 / (1+exp(-x)) Aktivierungsfunktion
        output_layer_act = tf.nn.sigmoid(output_layer)
        return output_layer_act
    
    def loss(self, y_true, y_pred):
        # 1 / N * sum((y_true - y_pred)^2)
        loss_func = tf.math.reduce_mean(tf.math.square(y_pred - y_true))
        self.current_loss_val = loss_func.numpy()
        return loss_func

    def update_variables(self, x_train, y_train):
        with tf.GradientTape() as tape:
            y_pred = self.predict(x_train)
            loss = self.loss(y_train, y_pred)
        gradients = tape.gradient(loss, self.variables)
        self.optimizer.apply_gradients(zip(gradients, self.variables))
        return loss

    def compute_metrics(self, x, y):
        y_pred = self.predict(x)
        y_pred_class = tf.reshape(tf.cast(tf.math.greater(y_pred, 0.5), tf.float32), y.shape) # wenn größer 0.5 dann Klasse 1 sonst Klasse 0
        correct_result = tf.math.equal(y_pred_class , y) # Vergleich mit wahren Werten
        accuracy = tf.math.reduce_mean(tf.cast(correct_result, tf.float32)) # Mittelwert bilden ( wie oft lag ich richtig ) -> Genauigkeit
        return accuracy

    def fit(self, x_train, y_train, epochs=10):
        print("Weights at the start: ", self.get_variables())

        for epoch in range(epochs):
            train_loss = self.update_variables(x_train, y_train).numpy()
            train_accuracy = self.compute_metrics(x_train, y_train).numpy()
            if epoch % 10 == 0:
                print("Epoch: ", epoch+1, " of ", epochs, 
                      " - Train loss: ", round(train_loss, 4),
                      " - Train Accuracy: ", round(train_accuracy, 4))
        print("Weights at the end: ", self.get_variables())

    def evaluate(self, x, y):
        loss = self.loss(self.predict(x), y).numpy()
        accuracy = self.compute_metrics(x, y).numpy()
        print("Loss: ", round(loss, 4), " Accuracy: ", round(accuracy, 4))


model = Model()
model.fit(x_train, y_train, epochs=epochs)
model.evaluate(x_test, y_test)