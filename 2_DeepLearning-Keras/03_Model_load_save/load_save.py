import os

from tensorflow.keras.models import load_model

# Save path
dir_path = os.path.abspath("../DeepLearning/models") # Linux und Windows
model_path_name = os.path.join(dir_path, "model.h5")

model.save_weights(filepath=model_path_name) # speichern
model.load_weights(filepath=model_path_name) # laden

model.save("model.h5") # speichern
model = load_model('model.h5') # laden

