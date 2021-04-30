""""""

import numpy as np
import tensorflow as tf

# %% np.absolute 与 tf.abs
x = np.array([-1.2, 1.2])
y = np.absolute(x)
print(y)

x = tf.constant(-1)
y = tf.constant([-1, 2])
sess = tf.Session()
print(sess.run(tf.abs(x)))  # 1
print(sess.run(tf.abs(y)))  # [1 2]

# %% np.max 与 tf.reduce_max
max_value = tf.reduce_max([1, 3, 2])
with tf.Session() as sess:
    max_value = sess.run(max_value)
    print(max_value)

x = np.array([1, 3, 2])
y = np.max(x)
print(y)
