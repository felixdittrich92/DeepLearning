import tensorflow as tf

# fix CuDnn problem
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
  except RuntimeError as e:
    print(e)
    
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

def build_cnn():
    # Define the CNN
    model = Sequential()
    # Conv Block 1
    model.add(Conv2D(32, (7,7), input_shape=(28,28,1)))
    model.add(Conv2D(32, (5,5)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation("relu"))
    # Conv Block 2
    model.add(Conv2D(64, (5,5)))
    model.add(Conv2D(128, (3,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation("relu"))
    # Fully connected layer 1
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    # Fully connected layer 1
    model.add(Dense(256))
    model.add(Activation("relu"))
    # Output layer
    model.add(Dense(10))
    model.add(Activation("softmax"))
    # Print the CNN layers
    model.summary()
    # Model object
    img = Input(shape=(28,28,1))
    pred = model(img)
    return Model(inputs=img, outputs=pred)
