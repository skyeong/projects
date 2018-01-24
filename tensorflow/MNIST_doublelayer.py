import tensorflow as tf
import numpy as np


num_units1 = 2
num_units2 = 2

x = tf.placeholder(tf.float32, [None, 2])

# Hidden Layer 1
w1 = tf.Variable(tf.truncated_normal([2, num_units1]))
b1 = tf.Variable(tf.zeros([num_units1]))
hidden1 = tf.nn.tanh(tf.matmul(x, w1)+b1)

# Hidden Layer2
w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
b2 = tf.Variable(tf.zeros([num_units2]))
hidden2 = tf.nn.tanh(tf.matmul(hidden1, w2) + b2)

# Output Layer
w0 = tf.Variable(tf.zeros([num_units2, 1]))
b0 = tf.Variable(tf.zeros([1])
# p = tf.nn.sigmoid(tf.matmul(hidden2, w0)+b0)


def generate_datablock(n, mu, var, t):
    data = multivariate_normal(mu, np.eye(2)*var, n)
    df = DataFrame(data, columns=['x1','x2'])
    df['t'] = t
    return df


df0 = generate_datablock(30, [-7, -7], 18, 1)
df1 = generate_datablock(30, [-7,7],  18, 0)
df2 = generate_datablock(30, [7,-7],  18, 0)
df3 = generate_datablock(30, [7,7],  18, 0)


df = pd.concat([df0, df1, df2, df3], ignore_index=True)
train_set = df.reindex(permutation(df.index)).reset_index(drop=True)
