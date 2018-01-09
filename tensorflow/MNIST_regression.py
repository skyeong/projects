import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Set random seed
np.random.seed(20180110)

# Load dataset
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)
images, labels = mnist.train.next_batch(10)

# Plot loaded data
fig = plt.figure(figsize=(8,4))
for c, (image, label) in enumerate(zip(images, labels)):
    subplot = fig.add_subplot(2,5,c+1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(label))
    subplot.imshow(image.reshape((28,28)), vmin=0, vmax=1, cmap=plt.cm.gray_r, interpolation="nearest")


# Define training dataset
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))  # 
w0 = tf.Variable(tf.zeros([1, 10]))
f = tf.matmul(x,w) + w0
p = tf.nn.softmax(f)

# Loss Function 
t = tf.placeholder(tf.float32, [None, 10])
loss = -tf.reduce_sum(t*tf.log(p))

# Define training step
train_step = tf.train.AdamOptimizer().minimize(loss)

# Accuracy function
correct_prediction = tf.equal(tf.argmax(p,1), tf.argmax(t,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Define Session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Training!
i=0
images, labels = mnist.test.images, mnist.test.labels
for _ in range(10000):
    i += 1
    batch_xs, batch_ts = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, t:batch_ts})
    if i%500 == 0:
        loss_val, acc_val = sess.run([loss,accuracy], feed_dict={x:images, t:labels})
        print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))


# Test
p_val = sess.run(p, feed_dict={x:images, t:labels})

fig = plt.figure(figsize=(8,15))
for i in range(10):
    c = 1
    for (image, label, pred) in zip(images, labels, p_val):
        prediction, actual = np.argmax(pred), np.argmax(label)
        if prediction != i:
            continue
        if (c<4 and i == actual) or (c>=4 and i!= actual):
            subplot = fig.add_subplot(10,6,i*6+c)
            subplot.set_xticks([])
            subplot.set_yticks([])
            subplot.set_title('%d / %d' % (prediction, actual))
            subplot.imshow(image.reshape((28,28)), vmin=0, vmax=1, cmap=plt.cm.gray_r, interpolation="nearest")
            c += 1
            if c>6:
                break