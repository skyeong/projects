""" Solution for assignment 2b - Task 1
logistic regression model for MNIST with placeholder
MNIST dataset: yann.lecun.com/exdb/mnist/
Target accuarcy >=0 0.97
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

import utils

# Define paramaters for the model
learning_rate = 0.001
batch_size = 128  # 128
n_epochs = 150

n_nodes_hl1 = 400    #400
n_nodes_hl2 = 150   # 150
n_nodes_hl3 = 80
# n_nodes_hl4 = 100
n_classes = 10

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
mnist = input_data.read_data_sets('data/mnist', one_hot=True)
X_batch, Y_batch = mnist.train.next_batch(batch_size)

# Step 2: create placeholders for features and labels
X = tf.placeholder(tf.float32, [batch_size, 784], name='image') 
Y = tf.placeholder(tf.int32, [batch_size, n_classes], name='label')

# Step 3: create weights and bias
hl1 = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
					'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
hl2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
hl3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
# hl4 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
# 					'biases':tf.Variable(tf.random_normal([n_nodes_hl4]))}
output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
				'biases':tf.Variable(tf.random_normal([n_classes]))}


# Step 4: build model
l1 = tf.add(tf.matmul(X, hl1['weights']), hl1['biases'])
l1 = tf.nn.sigmoid(l1)

l2 = tf.add(tf.matmul(l1, hl2['weights']), hl2['biases'])
l2 = tf.nn.sigmoid(l2)

l3 = tf.add(tf.matmul(l2, hl3['weights']), hl3['biases'])
l3 = tf.nn.sigmoid(l3)

# l4 = tf.add(tf.matmul(l3, hl4['weights']), hl4['biases'])
# l4 = tf.nn.softmax(l4)

logits = tf.matmul(l3, output_layer['weights'] + output_layer['biases'])


# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(entropy) # computes the mean over all the examples in the batch

# Step 6: define training op
# using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Step 7: calculate accuracy with test set
prediction = tf.nn.softmax(logits)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg_placeholder', tf.get_default_graph())
with tf.Session() as sess:
	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = int(mnist.train.num_examples/batch_size)
	
	# train the model n_epochs times
	for i in range(n_epochs): 
		total_loss = 0

		for j in range(n_batches):
			X_batch, Y_batch = mnist.train.next_batch(batch_size)
			_, loss_batch = sess.run([optimizer, loss], {X: X_batch, Y:Y_batch}) 
			total_loss += loss_batch
		print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
	print('Total time: {0} seconds'.format(time.time() - start_time))

	# test the model
	n_batches = int(mnist.test.num_examples/batch_size)
	total_correct_preds = 0

	for i in range(n_batches):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		accuracy_batch = sess.run(accuracy, {X: X_batch, Y:Y_batch})
		total_correct_preds += accuracy_batch	

	print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))

writer.close()