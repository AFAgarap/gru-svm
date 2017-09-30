# A Neural Network Architecture Combining Gated Recurrent Unit (GRU) and
# Support Vector Machine (SVM) for Intrusion Detection in Network Traffic Data
# Copyright (C) 2017  Abien Fred Agarap
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

"""Implementation of the GRU+SVM model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.3.7'
__author__ = 'Abien Fred Agarap'

import numpy as np
import os
import sys
import tensorflow as tf
import time


class GruSvm:

    def __init__(self, alpha, batch_size, cell_size, dropout_rate, num_classes, sequence_length, svm_c):
        self.alpha = alpha
        self.batch_size = batch_size
        self.cell_size = cell_size
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.svm_c = svm_c

        def __graph__():
            """Build the inference graph"""
            with tf.name_scope('input'):
                # [BATCH_SIZE, SEQUENCE_LENGTH]
                x_input = tf.placeholder(dtype=tf.uint8, shape=[None, self.sequence_length], name='x_input')

                # [BATCH_SIZE, SEQUENCE_LENGTH, 10]
                x_onehot = tf.one_hot(indices=x_input, depth=10, on_value=1.0, off_value=0.0, name='x_onehot')

                # [BATCH_SIZE]
                y_input = tf.placeholder(dtype=tf.uint8, shape=[None], name='y_input')

                # [BATCH_SIZE, N_CLASSES]
                y_onehot = tf.one_hot(indices=y_input, depth=self.num_classes, on_value=1.0, off_value=-1.0,
                                      name='y_onehot')

            state = tf.placeholder(dtype=tf.float32, shape=[None, self.cell_size], name='initial_state')

            p_keep = tf.placeholder(dtype=tf.float32, name='p_keep')
            learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

            cell = tf.contrib.rnn.GRUCell(self.cell_size)
            drop_cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=p_keep)

            # outputs: [BATCH_SIZE, SEQUENCE_LENGTH, CELL_SIZE]
            # states: [BATCH_SIZE, CELL_SIZE]
            outputs, states = tf.nn.dynamic_rnn(drop_cell, x_onehot, initial_state=state, dtype=tf.float32)

            states = tf.identity(states, name='H')

            with tf.name_scope('final_training_ops'):
                with tf.name_scope('weights'):
                    weight = tf.get_variable('weights',
                                             initializer=tf.random_normal([self.cell_size, self.num_classes],
                                                                          stddev=0.01))
                    self.variable_summaries(weight)
                with tf.name_scope('biases'):
                    bias = tf.get_variable('biases', initializer=tf.constant(0.1, shape=[self.num_classes]))
                    self.variable_summaries(bias)
                hf = tf.transpose(outputs, [1, 0, 2])
                last = tf.gather(hf, int(hf.get_shape()[0]) - 1)
                with tf.name_scope('Wx_plus_b'):
                    output = tf.matmul(last, weight) + bias
                    tf.summary.histogram('pre-activations', output)

            # L2-SVM
            with tf.name_scope('svm'):
                regularization_loss = 0.5 * tf.reduce_sum(tf.square(weight))
                hinge_loss = tf.reduce_sum(
                    tf.square(tf.maximum(tf.zeros([self.batch_size, self.num_classes]), 1 - y_onehot * output)))
                with tf.name_scope('loss'):
                    loss = regularization_loss + self.svm_c * hinge_loss
            tf.summary.scalar('loss', loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

            with tf.name_scope('accuracy'):
                predicted_class = tf.sign(output)
                predicted_class = tf.identity(predicted_class, name='prediction')
                with tf.name_scope('correct_prediction'):
                    correct = tf.equal(tf.argmax(predicted_class, 1), tf.argmax(y_onehot, 1))
                with tf.name_scope('accuracy'):
                    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            tf.summary.scalar('accuracy', accuracy)

            # merge all the summaries collected from the TF graph
            merged = tf.summary.merge_all()

            # set class properties
            self.x_input = x_input
            self.y_input = y_input
            self.y_onehot = y_onehot
            self.p_keep = p_keep
            self.loss = loss
            self.optimizer = optimizer
            self.state = state
            self.states = states
            self.learning_rate = learning_rate
            self.predicted_class = predicted_class
            self.accuracy = accuracy
            self.merged = merged

        sys.stdout.write('\n<log> Building Graph...')
        __graph__()
        sys.stdout.write('</log>\n')

    def train(self, checkpoint_path, log_path, model_name, epochs, train_data, train_size, validation_data,
              validation_size, result_path):
        """Train the model"""

        if not os.path.exists(path=checkpoint_path):
            os.mkdir(path=checkpoint_path)

        saver = tf.train.Saver(max_to_keep=1000)

        # initialize H (current_state) with values of zeros
        current_state = np.zeros([self.batch_size, self.cell_size])

        # variables initializer
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # get the time tuple
        timestamp = str(time.asctime())

        train_writer = tf.summary.FileWriter(logdir=log_path + timestamp + '-training', graph=tf.get_default_graph())
        validation_writer = tf.summary.FileWriter(logdir=log_path + timestamp + '-validation',
                                                  graph=tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(init_op)

            checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

            if checkpoint and checkpoint.model_checkpoint_path:
                saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

            try:
                for step in range(epochs * train_size // self.batch_size):

                    # set the value for slicing
                    # e.g. step = 0, batch_size = 256, train_size = 1898240
                    # (0 * 256) % 1898240 = 0
                    # [offset:(offset + batch_size)] = [0:256]
                    offset = (step * self.batch_size) % train_size
                    train_example_batch = train_data[0][offset:(offset + self.batch_size)]
                    train_label_batch = train_data[1][offset:(offset + self.batch_size)]

                    # dictionary for key-value pair input for training
                    feed_dict = {self.x_input: train_example_batch, self.y_input: train_label_batch,
                                 self.state: current_state,
                                 self.learning_rate: self.alpha, self.p_keep: self.dropout_rate}

                    train_summary, _, predictions, next_state = sess.run([self.merged, self.optimizer,
                                                                          self.predicted_class, self.states],
                                                                         feed_dict=feed_dict)

                    # Display training loss and accuracy every 100 steps and at step 0
                    if step % 100 == 0:
                        # get train loss and accuracy
                        train_loss, train_accuracy = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)

                        # display train loss and accuracy
                        print('step [{}] train -- loss : {}, accuracy : {}'.format(step, train_loss, train_accuracy))

                        # write the train summary
                        train_writer.add_summary(train_summary, step)

                        # save the model at current step
                        saver.save(sess, checkpoint_path + model_name, global_step=step)

                    current_state = next_state

                    predictions = sess.run(tf.argmax(predictions, 1))
                    predictions[predictions == 0] = -1
                    train_label_batch[train_label_batch == 0] = -1

                    self.save_labels(predictions=predictions, actual=train_label_batch, result_path=result_path,
                                     step=step)

                    # # concatenate the predicted labels and actual labels
                    # prediction_and_actual = np.concatenate((predictions, train_label_batch), axis=1)
                    #
                    # # save every prediction_and_actual numpy array to a CSV file for analysis purposes
                    # np.savetxt(os.path.join(result_path, 'gru_svm-{}-training.csv'.format(step)),
                    #            X=prediction_and_actual, fmt='%.3f', delimiter=',', newline='\n')

                for step in range(epochs * validation_size // self.batch_size):

                    offset = (step * self.batch_size) % validation_size
                    test_example_batch = validation_data[0][offset:(offset + self.batch_size)]
                    test_label_batch = validation_data[1][offset:(offset + self.batch_size)]

                    # dictionary for key-value pair input for validation
                    feed_dict = {self.x_input: test_example_batch, self.y_input: test_label_batch,
                                 self.state: np.zeros([self.batch_size, self.cell_size]), self.p_keep: 1.0}

                    validation_summary, predictions, validation_loss, validation_accuracy = \
                        sess.run([self.merged, self.predicted_class, self.loss, self.accuracy], feed_dict=feed_dict)

                    # Display validation loss and accuracy every 100 steps
                    if step % 100 == 0 and step > 0:
                        # get validation loss and accuracy
                        # validation_summary, validation_loss, validation_accuracy = sess.run([self.merged, self.loss,
                        #                                                                      self.accuracy],
                        #                                                                     feed_dict=feed_dict)

                        validation_writer.add_summary(validation_summary, step)

                        # display validation loss and accuracy
                        print('step [{}] validation -- loss : {}, accuracy : {}'.format(step, validation_loss,
                                                                                        validation_accuracy))
                    predictions = sess.run(tf.argmax(predictions, 1))
                    predictions[predictions == 0] = -1
                    test_label_batch[test_label_batch == 0] = -1

                    self.save_labels(predictions=predictions, actual=test_label_batch, result_path=result_path,
                                     step=step)

            except KeyboardInterrupt:
                print('Training interrupted at {}'.format(step))
            finally:
                print('EOF -- training done at step {}'.format(step))

            saver.save(sess, checkpoint_path + model_name, global_step=step)

    @staticmethod
    def variable_summaries(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    @staticmethod
    def save_labels(predictions, actual, result_path, step):
        """Saves the actual and predicted labels to a CSV file"""

        # concatenate the predicted labels and actual labels
        prediction_and_actual = np.array([predictions, actual], dtype=np.int8)

        # save every prediction_and_actual numpy array to a CSV file for analysis purposes
        np.savetxt(os.path.join(result_path, 'gru_svm-{}-training.csv'.format(step)),
                   X=prediction_and_actual, fmt='%.3f', delimiter=',', newline='\n')
