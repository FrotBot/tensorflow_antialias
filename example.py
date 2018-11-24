

import tensorflow as tf

a = tf.constant(3.0)
b = tf.constant(2.0)

c = a + b

sess = tf.Session()

print(sess.run(c))

