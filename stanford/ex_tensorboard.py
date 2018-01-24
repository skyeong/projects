import tensorflow as tf 
# a = tf.placeholder(tf.float32, shape=[3])
# b = tf.constant([3,3,3],tf.float32)
# c = a+b

# with tf.Session() as sess:
#     print(sess.run(c,{a:[1,2,3]}))

# writer = tf.summary.FileWriter('graphs/placeholders',tf.get_default_graph())


x = tf.Variable(10,name='x')
y = tf.Variable(20,name='y')
z = tf.add(x,y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('graphs/normal_loading', sess.graph)
    for _ in range(10):
        # sess.run(tf.add(x,y))
        sess.run(z)
    writer.close()
