import tensorflow as tf

x, y = 3,5
add_op = tf.add(x,y)
mul_op = tf.multiply(x,y)
useless = tf.multiply(x, add_op)
pow_op = tf.pow(add_op, mul_op)

# tensorboard에 point라는 이름으로 표시됨
op_summary = tf.summary.scalar('pow_op', pow_op)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('/Users/skyeong/pythonwork/tensorflow/mnist_sl_logs', sess.graph)

    result = sess.run([merged])
    sess.run(tf.global_variables_initializer())

    writer.add_summary(result[0])
