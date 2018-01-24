from IPython import get_ipython
get_ipython().magic('reset -sf')

import tensorflow as tf 


# a = tf.constant([2,2], name='vector')
# b = tf.constant([[0,1],[2,3]], name='matrix')
# x = tf.add(a,b, name='add')
# writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
# with tf.Session() as sess:
#     #print(sess.run(x))
#     print(sess.run(tf.div(b,a)))  # output: integer
#     print(sess.run(tf.divide(b,a)))  # output: float
#     print(sess.run(tf.truediv(b,a)))
#     print(sess.run(tf.floordiv(b,a)))

    
# writer.close()



a = tf.constant([10,20], name='a')
b = tf.constant([2,3], name='b')

with tf.Session() as sess:
    print(sess.run(tf.multiply(a,b)))
    print(sess.run(tf.tensordot(a,b,1)))  # element by element multiplication


