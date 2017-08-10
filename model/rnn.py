# Copyright 2017 Abien Fred Agarap. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/home/darth/Projects/Artificial Intelligence/tensorflow/tutorial/MNIST_data',
                                  one_hot=True)

hm_epochs = 10
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 32
svmc = 0.5

x = tf.placeholder(dtype=tf.float32, shape=[None, n_chunks, chunk_size])
y = tf.placeholder(dtype=tf.float32)
Hin = tf.placeholder(dtype=tf.float32, shape=[None, rnn_size * 3])


def recurrent_neural_network(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes], stddev=0.01)),
             'biases': tf.Variable(tf.constant(0.1, shape=[n_classes]))}

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)

    # cells = [tf.contrib.rnn.GRUCell(rnn_size) for _ in range(3)]
    cell = tf.contrib.rnn.GRUCell(rnn_size)
    # dropcells = [tf.contrib.rnn.DropoutWrapper(cell, 0.8) for cell in cells]
    # mcell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=False)
    # mcell = tf.contrib.rnn.DropoutWrapper(mcell, 0.8)
    # outputs, states = tf.nn.dynamic_rnn(mcell, inputs=x, initial_state=Hin, dtype=tf.float32)
    outputs, states = tf.nn.static_rnn(cell, x, dtype=tf.float32)
    # outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output, layer


def train_neural_network(x):
    prediction, layer = recurrent_neural_network(x)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    regularization_loss = 0.5 * tf.reduce_sum(tf.square(layer['weights']))
    hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([batch_size, n_classes]), 1 - tf.cast(y, tf.float32) * prediction))
    cost = regularization_loss + 0.5 * hinge_loss
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    # optimizer = tf.train.AdagradOptimizer(0.1).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_y[epoch_y == 0] = -1
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})

                epoch_loss += c

            print('Epoch : {} completed out of {}, loss : {}'.format(epoch, hm_epochs, epoch_loss))

        # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        predicted_class = tf.sign(prediction)
        # correct = tf.equal(y, predicted_class)
        correct = tf.equal(tf.argmax(predicted_class, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        x_ = mnist.test.images.reshape((-1, n_chunks, chunk_size))
        y_ = mnist.test.labels
        y_[y_ == 0] = -1

        accuracy_, predicted_class_, y_ = sess.run([accuracy, tf.argmax(predicted_class, 1), tf.argmax(y, 1)],
                                                   feed_dict={x: x_, y: y_})

        # for index in range(len(y_)):
        #     print('{} -- y : {}, y^ : {}'.format(index, y_[index], predicted_class_[index]))
        print('Accuracy : {}'.format(accuracy_))


if __name__ == '__main__':
    train_neural_network(x)
