import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices(tf.zeros([10, 400, 400, 3]))

for item in dataset:
    print(item.shape)