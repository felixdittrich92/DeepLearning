'''
Generative Adversarial Attacks - den Output des Netes beeinflussen 
(z.B. andere Klasse ausgeben als eigentlich ist) durch Noise über ganzes Bild was für das menschliche Auge nicht sichtbar ist,
oder anbringen von Patches (kleine Bilder die dann im Vordergrund liegen) welche dann eine
z.B. andere Klassifizierung bewirken.

untargeted - ohne bestimmte "erzwungene" Klasse
'''
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

import numpy as np
import matplotlib.pyplot as plt

from mnistCnn import *
from mnistData import *
from plotting import *

mnist_data = MNIST()
x_train, y_train = mnist_data.get_train_set()
x_test, y_test = mnist_data.get_test_set()

def adversarial_noise(model, image, label):
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image, training=False)[0]
        loss = tf.keras.losses.categorical_crossentropy(label, prediction)
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, image)
    # Get the sign of the gradients to create the noise
    signed_grad = tf.sign(gradient)
    return signed_grad

if __name__ == "__main__":
    cnn = build_cnn()

    lr = 0.0001
    optimizer = Adam(lr=lr)
    cnn.compile(
        loss="categorical_crossentropy", 
        optimizer=optimizer, 
        metrics=["accuracy"])

    #cnn.fit(x_train, y_train, verbose=1,
    #        batch_size=256, epochs=10,
    #        validation_data=(x_test, y_test))
    path = "/home/felix/Desktop/DeepLearning/7_DeepLearning-GANs/04_Generative_Adversarial_Attacks/weights/mnist_cnn.h5"
    #cnn.save_weights(path)
    cnn.load_weights(path)
    #score = cnn.evaluate(x_test, y_test, verbose=0)
    #print("Test accuracy: ", score[1])

    sample_idx = np.random.randint(low=0, high=x_test.shape[0])
    image = np.array([x_test[sample_idx]])
    true_label = y_test[sample_idx]
    true_label_idx = np.argmax(true_label)
    y_pred = cnn.predict(image)[0]
    print("Right class: ", true_label_idx)
    print("Prob. right class: ", y_pred[true_label_idx])
    
    eps = 0.005 # Stärke des Noise filters pro Schritt
    image_adv = tf.convert_to_tensor(image, dtype=tf.float32) # Bild in Tensor umwandeln

    while (np.argmax(y_pred) == true_label_idx):
        # image_adv = image_adv + eps * noise
        noise = adversarial_noise(cnn, image_adv, true_label)
        if np.sum(noise) == 0.0:
            break
        image_adv = image_adv + eps * noise
        image_adv = tf.clip_by_value(image_adv, 0, 1)
        y_pred = cnn.predict(image_adv)[0]
        print("Prob. right class: ", y_pred[true_label_idx])
        print("Highest Prob.: ", np.max(y_pred), "\n")

    plot_img(image_adv.numpy(), cmap="gray")
    plot_img(noise.numpy(), cmap="gray")


