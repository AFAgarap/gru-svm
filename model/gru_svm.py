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

"""Implementation of the GRU+SVM model for Intrusion Detection"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.3.4'
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
DROPOUT_P_KEEP = 0.85
HM_EPOCHS = 4
N_CLASSES = 2
SEQUENCE_LENGTH = 21
SVM_C = 0.5

# learning rate decay parameters
LEARNING_RATE = 1e-5


class GruSvm:

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

            state = tf.placeholder(dtype=tf.float32, shape=[None, CELL_SIZE], name='initial_state')

            p_keep = tf.placeholder(dtype=tf.float32, name='p_keep')
            learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

            cell = tf.contrib.rnn.GRUCell(CELL_SIZE)
            drop_cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=p_keep)

            # outputs: [BATCH_SIZE, SEQUENCE_LENGTH, CELL_SIZE]
            # states: [BATCH_SIZE, CELL_SIZE]
            outputs, states = tf.nn.dynamic_rnn(drop_cell, x_input, initial_state=state, dtype=tf.float32)

            states = tf.identity(states, name='H')

            with tf.name_scope('final_training_ops'):
                with tf.name_scope('weights'):
                    weight = tf.get_variable('weights',
                                             initializer=tf.random_normal([BATCH_SIZE, N_CLASSES], stddev=0.01))
                    self.variable_summaries(weight)
                with tf.name_scope('biases'):
                    bias = tf.get_variable('biases', initializer=tf.constant(0.1, shape=[N_CLASSES]))
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
                    tf.square(tf.maximum(tf.zeros([BATCH_SIZE, N_CLASSES]), 1 - y_input * output)))
                with tf.name_scope('loss'):
                    loss = regularization_loss + SVM_C * hinge_loss
            tf.summary.scalar('loss', loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

            with tf.name_scope('accuracy'):
                predicted_class = tf.sign(output)
                predicted_class = tf.identity(predicted_class, name='prediction')
                with tf.name_scope('correct_prediction'):
                    correct = tf.equal(tf.argmax(predicted_class, 1), tf.argmax(y_input, 1))
                with tf.name_scope('accuracy'):
                    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            tf.summary.scalar('accuracy', accuracy)

            # merge all the summaries collected from the TF graph
            merged = tf.summary.merge_all()

            # set class properties
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

    def train(self, train_data, validation_data, result_path):
        """Train the model"""

        if not os.path.exists(path=self.checkpoint_path):
            os.mkdir(path=self.checkpoint_path)

        saver = tf.train.Saver(max_to_keep=1000)

        # initialize H (current_state) with values of zeros
        current_state = np.zeros([BATCH_SIZE, CELL_SIZE])

        # variables initializer
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # get the time tuple
        timestamp = str(time.asctime())

        train_writer = tf.summary.FileWriter(logdir=self.log_path + timestamp + '-training', graph=tf.get_default_graph())
        validation_writer = tf.summary.FileWriter(logdir=self.log_path + timestamp + '-validation',
                                                  graph=tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(init_op)

            checkpoint = tf.train.get_checkpoint_state(self.checkpoint_path)

            if checkpoint and checkpoint.model_checkpoint_path:
                saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
                saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint_path))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                step = 0
                while not coord.should_stop():
                    train_example_batch, train_label_batch = sess.run([train_data[0], train_data[1]])

                    # dictionary for key-value pair input for training
                    feed_dict = {self.x_input: train_example_batch, self.y_input: train_label_batch,
                                 self.state: current_state,
                                 self.learning_rate: LEARNING_RATE, self.p_keep: DROPOUT_P_KEEP}

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
                        saver.save(sess, self.checkpoint_path + self.model_name, global_step=step)

                    # Display validation loss and accuracy every 100 steps
                    if step % 100 == 0 and step > 0:
                        # retrieve validation data
                        test_example_batch, test_label_batch = sess.run([validation_data[0],
                                                                         validation_data[1]])
                        # dictionary for key-value pair input for validation
                        feed_dict = {self.x_input: test_example_batch, self.y_input: test_label_batch,
                                     self.state: np.zeros([BATCH_SIZE, CELL_SIZE]), self.p_keep: 1.0}

                        # get validation loss and accuracy
                        validation_summary, validation_loss, validation_accuracy = sess.run([self.merged, self.loss,
                                                                                             self.accuracy],
                                                                                            feed_dict=feed_dict)

                        validation_writer.add_summary(validation_summary, step)

                        # display validation loss and accuracy
                        print('step [{}] validation -- loss : {}, accuracy : {}'.format(step, validation_loss,
                                                                                        validation_accuracy))

                    current_state = next_state

                    # concatenate the predicted labels and actual labels
                    prediction_and_actual = np.concatenate((predictions, train_label_batch), axis=1)

                    # save every prediction_and_actual numpy array to a CSV file for analysis purposes
                    np.savetxt(os.path.join(result_path, 'gru_svm-{}-training.csv'.format(step)),
                               X=prediction_and_actual, fmt='%.3f', delimiter=',', newline='\n')

                    step += 1
            except tf.errors.OutOfRangeError:
                print('EOF -- training done at step {}'.format(step))
            except KeyboardInterrupt:
                print('Training interrupted at {}'.format(step))
            finally:
                train_writer.close()
                coord.request_stop()
            coord.join(threads)

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
    parser = argparse.ArgumentParser(description='GRU+SVM for Intrusion Detection')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-t', '--train_dataset', required=True, type=str,
                       help='path of the training dataset to be used')
    group.add_argument('-v', '--validation_dataset', required=True, type=str,
                       help='path of the validation dataset to be used')
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


def main(argv):

    # get the train data
    # features: train_data[0], labels: train_data[1]
    train_data = data.input_pipeline(path=argv.train_dataset, batch_size=BATCH_SIZE,
                                     num_classes=N_CLASSES, num_epochs=HM_EPOCHS)

    # get the validation data
    # features: validation_data[0], labels: validation_data[1]
    validation_data = data.input_pipeline(path=argv.validation_dataset, batch_size=BATCH_SIZE,
                                          num_classes=N_CLASSES, num_epochs=1)

    # instantiate the model
    model = GruSvm(checkpoint_path=argv.checkpoint_path, log_path=argv.log_path, model_name=argv.model_name)

    # train the model
    model.train(train_data=train_data, validation_data=validation_data, result_path=argv.result_path)


if __name__ == '__main__':
    args = parse_args()

    main(argv=args)
