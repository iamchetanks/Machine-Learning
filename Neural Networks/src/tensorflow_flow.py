# to ensure compatability with both Python 2 and 3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import model


# Parameter definitions
batch_size = 200
learning_rate = 0.05
max_steps = 1000



# Prepare data
X_train, Y_train = model.read_file("../data/downgesture_train.list")
X_test, Y_test = model.read_file("../data/downgesture_test.list")
#print(Y_train)

# input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, 960])
labels_placeholder = tf.placeholder(tf.int64, shape=[None],name="label")

# variables (these are the values we want to optimize)
w1 = tf.Variable(tf.random_normal([960, 100]) * 0.01)
b1 = tf.Variable(tf.random_normal([100]) * 0.01)
y1 = tf.matmul(images_placeholder, w1) + b1

w2 = tf.Variable(tf.random_normal([100,2]) * 0.01)
b2 = tf.Variable(tf.random_normal([2]) * 0.01)

# the classifier's result
logits = tf.matmul(y1, w2) + b2
#logits = tf.Print(tf.expand_dims([logits], 0), [logits])

#print (labels_placeholder.shape)
#loss=(logits[0])
# Define the loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels_placeholder, logits = logits))

# training operation
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Operation comparing prediction with true label
correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholder)

# Operation calculating the accuracy of our predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Repeat max_steps times
    for i in range(max_steps):
        # Generate input data batch
        indices = np.random.choice(X_train.T.shape[0], batch_size)
        images_batch = X_train.T[indices]
        labels_batch = Y_train.T[indices]
        #print(labels_batch.shape)
        sess.run(train_step, feed_dict={
            images_placeholder: images_batch, labels_placeholder: labels_batch.flatten()})

        if i % 100 == 0:
              train_accuracy = sess.run(accuracy, feed_dict={
              images_placeholder: images_batch, labels_placeholder: labels_batch.flatten()})
              print('{:5d}th training accuracy {:g}'.format(i, train_accuracy))


    test_accuracy = sess.run(accuracy, feed_dict={
        images_placeholder: X_test.T, labels_placeholder: Y_test.T.flatten()})
    print('Test accuracy {:g}'.format(test_accuracy))