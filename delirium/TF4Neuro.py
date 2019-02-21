import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import tensorflow as tf

# Set random seed
np.random.seed(20180111)
tf.set_random_seed(20180111)

# sys.path.append('/Users/skyeong/afni/')

class TF4Neuro:
    def __init__(self):
        self.__name__ = 'TF4Neuro'
        print ('TF4Neuro is created.')
        self.df = DataFrame()
        self.edges = list()
        self.learn_rate = 0.0001
        self.nNodes = 10

    def insert_data(self,fn_csv):
        #fn_csv = os.path.join(datapath,filename)
        df = pd.read_csv(fn_csv, sep=',', header=0)
        self.df = df.reindex(np.random.permutation(df.index)) # randomizing dataset
        print('shape = (%d, %d)' % self.df.shape)

    def set_train_x(self, edges, filter_key=None, filter_value=None):
        self.nFeatures = len(edges)
        self.edges = edges
        if filter_key is not None:
            df1 = self.df[self.df[filter_key]==filter_value]
        else:
            df1 = self.df
        self.train_x = df1[edges].as_matrix()

    def set_train_t(self,targetTag,filter_key=None, filter_value=None):
        if filter_key is not None:
            df1 = self.df[self.df[filter_key]==filter_value]
        else:
            df1 = self.df
        self.train_t = df1[targetTag].as_matrix().reshape([len(df1),1])

    def set_test_x(self, edges, filter_key=None, filter_value=None):
        self.nFeatures = len(edges)
        self.edges = edges
        if filter_key is not None:
            df1 = self.df[self.df[filter_key]==filter_value]
        else:
            df1 = self.df
        self.test_x = df1[edges].as_matrix()

    def set_test_t(self,targetTag,filter_key=None, filter_value=None):
        if filter_key is not None:
            df1 = self.df[self.df[filter_key]==filter_value]
        else:
            df1 = self.df
        self.test_t = df1[targetTag].as_matrix().reshape([len(df1),1])

    def set_num_nodes(self,nNodes):
        self.nNodes = nNodes

    def prepare_singlelayer_session(self):
        self.x = tf.placeholder(tf.float32, [None, self.nFeatures])
       
        # Parameters in Hidden layer
        w1 = tf.Variable(tf.truncated_normal([self.nFeatures, self.nNodes]))  # initialize using random normal distribution
        b1 = tf.Variable(tf.truncated_normal([self.nNodes]))
        hidden1 = tf.nn.sigmoid(tf.matmul(self.x,w1) + b1)  # *mult is to boost speed

        # Parameters in Output layer
        w0 = tf.Variable(tf.zeros([self.nNodes, 1]))
        b0 = tf.Variable(tf.zeros([1]))
        p = tf.nn.sigmoid(tf.matmul(hidden1,w0)+b0)

        # Error function
        self.t = tf.placeholder(tf.float32, [None,1])
        self.loss = -tf.reduce_sum(self.t*tf.log(p) + (1-self.t)*tf.log(1-p))
        self.train_step = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(self.loss)

        # Accuracy function
        correct_prediction = tf.equal(tf.sign(p-0.5), tf.sign(self.t-0.5))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Create Session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())


    # Training Step
    def run_training_step(self,niter):
        sess     = self.sess
        x        = self.x
        t        = self.t
        train_x  = self.train_x
        train_t  = self.train_t
        loss     = self.loss
        accuracy = self.accuracy

        # for i,ii in enumerate(range(niter)):
        #     sess.run(self.train_step, feed_dict={x:batch_x, t:batch_t})
        #     if (i%(niter/10) == 0):
        #         loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x:train_x, t:train_t})
<        #         print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val) )
        datapoint_size = len(train_x)
        batch_size = 20
        for i,ii in enumerate(range(niter)):
            sess.run(self.train_step, feed_dict={x:train_x, t:train_t})
            if (i%(niter/10) == 0):
                loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x:self.test_x, t:self.test_t})
                print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val) )



if __name__=="__main__":

    edges = list()
    for i in range(116):
        for j in range(i+1,116):
            varName = 'r%d_%d'%(i+1,j+1)
            edges.append(varName)

    nNodes = 2  # define number of nodes in hidden layer
    niter = 100000

    # Create TensorFlow Class
    neuro = TF4Neuro()
    neuro.insert_data('/Users/skyeong/data/delirium/Results/delirium_network_n116.csv')
    neuro.set_train_x(['r2_108','r4_92','r10_25','r10_26','r28_116','r32_60'],filter_key='expYear',filter_value=2014)
    neuro.set_train_t('Dx',filter_key='expYear',filter_value=2014)
    neuro.set_test_x(['r2_108','r4_92','r10_25','r10_26','r28_116','r32_60'],filter_key='expYear',filter_value=2008)
    neuro.set_test_t('Dx',filter_key='expYear',filter_value=2008)
    neuro.set_num_nodes(nNodes)
    neuro.prepare_singlelayer_session()
    neuro.run_training_step(niter)

   
