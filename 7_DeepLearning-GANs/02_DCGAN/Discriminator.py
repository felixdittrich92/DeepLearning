'''
Discriminator - bewertet die vom Generator erzeugten Bilder ob Real oder Fake
'''

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

def build_discriminator(img_shape):
    model = Sequential() #28x28
    
    model.add(Conv2D(64, kernel_size=5, strides=2, input_shape=img_shape, padding="same")) #14x14
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten()) #4x4x256 => 16 x 256 = 2^4 x 2^8 = 2^12 = 4096 
    model.add(Dense(1))
    model.add(Activation("sigmoid")) # < 0.5 Klasse 0  > 0.5 Klasse 1

    model.summary()
    img = Input(shape=img_shape)
    d_pred = model(img) # model auf bild aufrufen
    return Model(inputs=img, outputs=d_pred) # beeinhaltet komplettes Modell