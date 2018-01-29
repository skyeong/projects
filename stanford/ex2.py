import tensorflow as tf 

dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32),
                                            (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE,1])))
iterator = dataset.make_initializable_iterator()
center_words, target_words = iterator.get_next()

embed_matrix = tf.get_variable('embed_matrix',
                               shape=[VOCAB_SIZE, EMBED_SIZE],
                               initializer=tf.random_uniform_initializer())

embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')

nce_weight = tf.get_variable('nce_weight',
            shape=[VOCAB_SIZE, EMBED_SIZE],
            initializer=tf.truncated_normal_initializer(stdev=1.0 / (EMBED_SIZE ** 0.5)))
nce_bias = tf.get_variable('nce_bias', initializer=tf.zeros([VOCAB_SIZE]))

tf.matmul(embed, tf.transpose(nce_weight)) + nce_bias


loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                    biases=nce_bias,
                                    labels=target_words,
                                    inputs=embed,
                                    num_sampled=NUM_SAMPLED,
                                    num_classes=VOCAB_SIZE))

optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
