import os

from sklearn.model_selection import cross_val_score

from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from Dataset_class import MNIST

mnist = MNIST()
mnist.data_augmentation(augment_size=5000)
mnist.data_preprocessing(preprocess_mode="MinMax")
x_train, y_train = mnist.get_train_set()
x_test, y_test = mnist.get_test_set()
num_classes = mnist.num_classes

# Model 1: MinMax, 3 Conv Blocks, Dense: 128, LR=0.001
# Model 3: MinMax, 3 Conv Blocks, Dense: 256, LR=0.001

# Model params
lr = 0.001
optimizer = Adam(lr=lr)
epochs = 3
batch_size = 128

def model_fn():
    # Define the CNN
    input_img = Input(shape=x_train.shape[1:])

    x = Conv2D(filters=32, kernel_size=3, padding='same')(input_img)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=3, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(units=128)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(x)

    # Build the model
    model = Model(inputs=[input_img], outputs=[y_pred])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"])
    return model

keras_clf = KerasClassifier(
    build_fn=model_fn,
    epochs=epochs,
    batch_size=batch_size,
    verbose=0)

scores = cross_val_score(
    estimator=keras_clf,
    X=x_train,
    y=y_train,
    cv=3,
    n_jobs=1)

print("Score list: ", scores)
print("Mean Acc: %0.6f (+/- %0.6f)" % (np.mean(scores), np.std(scores)*2))
