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

"""Implementation of GRU+Softmax model for Intrusion Detection"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.3.2'
__author__ = 'Abien Fred Agarap'

import argparse
import data
import numpy as np
import os
import sys
import tensorflow as tf
import time

# hyper-parameters
BATCH_SIZE = 256
CELL_SIZE = 256
DROPOUT_P_KEEP = 0.8
HM_EPOCHS = 2
LEARNING_RATE = 1e-6
N_CLASSES = 2
SEQUENCE_LENGTH = 21


class GruSoftmax:

    def __init__(self, checkpoint_path, log_path, model_name):
        self.checkpoint_path = checkpoint_path
        self.log_path = log_path
        self.model_name = model_name

        def __graph__():
            """Build the inference graph"""
            with tf.name_scope('input'):
                # [BATCH_SIZE, SEQUENCE_LENGTH, 10]
                x_input = tf.placeholder(dtype=tf.float32, shape=[None, SEQUENCE_LENGTH, 10], name='x_input')

                # [BATCH_SIZE, N_CLASSES]
                y_input = tf.placeholder(dtype=tf.float32, shape=[None, N_CLASSES], name='y_input')

            # [BATCH_SIZE, CELL_SIZE]
            state = tf.placeholder(dtype=tf.float32, shape=[None, CELL_SIZE], name='initial_state')

            learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            p_keep = tf.placeholder(tf.float32, name='p_keep')

            cell = tf.contrib.rnn.GRUCell(CELL_SIZE)
            drop_cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=p_keep)

            # outputs: [BATCH_SIZE, SEQUENCE_LENGTH, CELL_SIZE]
            # states: [BATCH_SIZE, CELL_SIZE]
            outputs, states = tf.nn.dynamic_rnn(drop_cell, x_input, initial_state=state, dtype=tf.float32)

            states = tf.identity(states, name='H')

            with tf.name_scope('final_training_ops'):
                with tf.name_scope('weights'):
                    weight = tf.get_variable('weights',
                                             initializer=tf.random_normal([CELL_SIZE, N_CLASSES], stddev=0.01))
                    self.variable_summaries(weight)
                with tf.name_scope('biases'):
                    bias = tf.get_variable('biases', initializer=tf.constant(0.1, shape=[N_CLASSES]))
                    self.variable_summaries(bias)
                hf = tf.transpose(outputs, [1, 0, 2])
                last = tf.gather(hf, int(hf.get_shape()[0]) - 1)
                with tf.name_scope('Wx_plus_b'):
                    output = tf.matmul(last, weight) + bias
                    tf.summary.histogram('pre-activations', output)

            # Softmax
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_input))
            tf.summary.scalar('loss', loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

            with tf.name_scope('accuracy'):
                predicted_class = tf.nn.softmax(output)
                with tf.name_scope('correct_prediction'):
                    correct = tf.equal(tf.argmax(predicted_class, 1), tf.argmax(y_input, 1))
                with tf.name_scope('accuracy'):
                    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            tf.summary.scalar('accuracy', accuracy)

            merged = tf.summary.merge_all()  # merge all the summaries collected from TF graph

            self.x_input = x_input
            self.y_input = y_input
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

    def train(self, train_data, train_size, validation_data, validation_size, result_path):
        """Train the model"""
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        saver = tf.train.Saver(max_to_keep=1000)

        current_state = np.zeros([BATCH_SIZE, CELL_SIZE])  # initialize H (current_state) with values of zeros

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())  # variable initializer

        timestamp = str(time.asctime())  # get the time in seconds since the Epoch

        # create an event file to contain the TF graph summaries for training
        train_writer = tf.summary.FileWriter(self.log_path + timestamp + '-training', graph=tf.get_default_graph())

        # create an event file to contain the TF graph summaries for validation
        validation_writer = tf.summary.FileWriter(self.log_path + timestamp + '-validation', graph=tf.get_default_graph())

        with tf.Session() as sess:

            sess.run(init_op)

            checkpoint = tf.train.get_checkpoint_state(self.checkpoint_path)

            # check if a trained model exists
            if checkpoint and checkpoint.model_checkpoint_path:
                # load the graph of the trained model
                saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
                # restore variables to resume training
                saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint_path))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                step = 0
                while not coord.should_stop():
                    train_example_batch, train_label_batch = sess.run([self.train_data[0], self.train_data[1]])

                    # changed the range of labels for Softmax to {0, 1}
                    train_label_batch[train_label_batch == -1] = 0

                    # dictionary for key-value pair input for training
                    feed_dict = {self.x_input: train_example_batch, self.y_input: train_label_batch,
                                 self.state: current_state,
                                 self.learning_rate: LEARNING_RATE, self.p_keep: DROPOUT_P_KEEP}

                    summary, _, epoch_loss, predictions, next_state = sess.run([self.merged, self.optimizer, self.loss,
                                                                                self.predicted_class, self.states],
                                                                               feed_dict=feed_dict)

                    # Display training accuracy every 100 steps and at step 0
                    if step % 100 == 0:
                        accuracy_ = sess.run(self.accuracy, feed_dict=feed_dict)
                        print('step [{}] train -- loss : {}, accuracy : {}'.format(step, epoch_loss, accuracy_))
                        train_writer.add_summary(summary, step)
                        saver.save(sess, self.checkpoint_path + self.model_name, global_step=step)

                    # Validate training every 100 steps
                    if step % 100 == 0 and step > 0:
                        test_example_batch, test_label_batch = sess.run([self.test_data[0], self.test_data[1]])

                        # change the range of labels for Softmax to {0, 1}
                        test_label_batch[test_label_batch == -1] = 0

                        # dictionary for key-value pair input for validation
                        feed_dict = {self.x_input: test_example_batch, self.y_input: test_label_batch,
                                     self.state: np.zeros([BATCH_SIZE, CELL_SIZE]), self.p_keep: 1.0}

                        summary, test_loss, test_accuracy = sess.run([self.merged, self.loss, self.accuracy],
                                                                     feed_dict=feed_dict)

                        print('step [{}] validation -- loss : {}, accuracy : {}'.format(step, test_loss, test_accuracy))

                        validation_writer.add_summary(summary, step)

                    current_state = next_state

                    # concatenate the predicted labels and actual labels
                    prediction_and_actual = np.concatenate((predictions, train_label_batch), axis=1)

                    # save every prediction_and_actual numpy array to a CSV file for analysis purposes
                    np.savetxt(os.path.join(self.result_path, 'gru_softmax-{}-training.csv'.format(step)),
                               X=prediction_and_actual, fmt='%.3f', delimiter=',', newline='\n')

                    step += 1
            except tf.errors.OutOfRangeError:
                print('EOF -- training done at step {}'.format(step))
            except KeyboardInterrupt:
                print('Training interrupted at {}'.format(step))
            finally:
                train_writer.close()
                validation_writer.close()
                coord.request_stop()

            coord.join(threads)

            saver = tf.train.Saver()
            saver.save(sess, self.checkpoint_path + self.model_name, global_step=step)

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


def parse_args():
    parser = argparse.ArgumentParser(description='GRU+Softmax for Intrusion Detection')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-t', '--train_dataset', required=True, type=str,
                       help='the NumPy array training dataset (*.npy) to be used')
    group.add_argument('-v', '--validation_dataset', required=True, type=str,
                       help='the NumPy array validation dataset (*.npy) to be used')
    group.add_argument('-c', '--checkpoint_path', required=True, type=str,
                       help='path where to save the trained model')
    group.add_argument('-l', '--log_path', required=True, type=str,
                       help='path where to save the TensorBoard logs')
    group.add_argument('-m', '--model_name', required=True, type=str,
                       help='filename for the trained model')
    group.add_argument('-r', '--result_path', required=True, type=str,
                       help='path where to save the actual and predicted labels')
    arguments = parser.parse_args()
    return arguments


def main(arguments):

    # get the train data
    # features: train_data[0], labels: train_data[1]
    train_features, train_labels = data.load_data(dataset=arguments.train_dataset)

    # get the validation data
    # features: validation_data[0], labels: validation_data[1]
    validation_features, validation_labels = data.load_data(dataset=arguments.validation_dataset)

    # get the size of the dataset for slicing
    train_size = train_features.shape[0]
    validation_size = validation_features.shape[0]

    # slice the dataset to be exact as per the batch size
    # e.g. train_size = 1898322, batch_size = 256
    # [:1898322-(1898322%256)] = [:1898240]
    # 1898322 // 256 = 7415; 7415 * 256 = 1898240
    train_features = train_features[:train_size-(train_size % BATCH_SIZE)]
    train_labels = train_labels[:train_size-(train_size % BATCH_SIZE)]

    # modify the size of the dataset to be passed on model.train()
    train_size = train_features.shape[0]

    # slice the dataset to be exact as per the batch size
    validation_features = validation_features[:validation_size-(validation_size % BATCH_SIZE)]
    validation_labels = validation_labels[:validation_size-(validation_size % BATCH_SIZE)]

    # modify the size of the dataset to be passed on model.train()
    validation_size = validation_features.shape[0]

    model = GruSoftmax(checkpoint_path=arguments.checkpoint_path, log_path=arguments.log_path,
                       model_name=arguments.model_name)

    model.train(train_data=[train_features, train_labels], train_size=train_size,
                validation_data=[validation_features, validation_labels], validation_size=validation_size,
                result_path=arguments.result_path)


if __name__ == '__main__':
    args = parse_args()

    main(args)
