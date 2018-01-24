import tensorflow as tf 

# s = tf.get_variable('scalar',initializer=tf.constant(2))
# m = tf.get_variable('matrix',initializer=tf.constant([[0,1],[2,3]]))
# W = tf.get_Variable('big_matrix',initializer=tf.zeros_initializer())
# V = tf.get_variable("normal_matrix",shape=(784,10),initializer=tf.truncated_normal_initializer())

# print(tf.Session().run(tf.report_uninitialized_variables()))

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
    # print(V.eval())


# W = tf.Variable(10)
# W.assign(100)
# with tf.Session() as sess:
#     sess.run(W.initializer)
#     print(W.eval())


# a = tf.get_variable('scalar',initializer=tf.constant(2))
# a_times_two = a.assign(a*2)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(a_times_two))
#     print(sess.run(a_times_two))
#     print(sess.run(a_times_two))


# W = tf.get_variable('scalar',initializer=tf.constant(10))
# with tf.Session() as sess:
#     sess.run(W.initializer)
#     print("W: ", sess.run(W.assign_add(10)))
#     print("W: ", sess.run(W.assign_sub(2)))

# Tensorflow sessions maintain values sepqrately
W = tf.get_variable('scalar',initializer=tf.constant(10))
sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(W.initializer)
sess2.run(W.initializer)

print(sess1.run(W.assign_add(10)))
print(sess2.run(W.assign_sub(2 )))
print(sess1.run(W.assign_add(100)))
print(sess2.run(W.assign_sub(50)))

sess1.close()
sess2.close()