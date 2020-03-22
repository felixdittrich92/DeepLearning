import tensorflow as tf

value = tf.test.is_gpu_available()
print(value)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print("hello")
print("hello Ubuntu")