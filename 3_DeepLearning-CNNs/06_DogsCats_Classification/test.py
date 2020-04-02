import numpy as np
from matplotlib import pyplot as plt

img_array = np.load('../DeepLearning/data/PetImages/x.npy')

plt.imshow(img_array[20000], cmap='BrBG')
plt.show()

label = np.load('../DeepLearning/data/PetImages/y.npy')
print(label[20000])