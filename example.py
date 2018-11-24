

import tensorflow as tf

a = tf.placeholder(tf.float32, shape=(2,1))
b = tf.placeholder(tf.float32, shape=(1,2))


c = tf.matmul(a,b)

sess = tf.Session()

print(sess.run(c, {a: [[1],[2]], b: [[3,4]]}))
