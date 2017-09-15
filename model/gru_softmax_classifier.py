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

"""Classifier program based on the GRU+Softmax model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.3.4'
__author__ = 'Abien Fred Agarap'

import argparse
import numpy as np
import tensorflow as tf

BATCH_SIZE = 256
CELL_SIZE = 256
DROPOUT_P_KEEP = 1.0
NUM_BIN = 10
NUM_CLASSES = 2


def predict(test_path, checkpoint_path, result_filename):
    """Classifies the data whether there is an attack or none"""

    test_file = test_path + '/24.csv'
    # load the CSV file to numpy array
    test_example_batch = np.genfromtxt(test_file, delimiter=',')

    # isolate the label to a different numpy array
    test_label_batch = test_example_batch[:, 17]

    # cast the label array to float32
    test_label_batch = test_label_batch.astype(np.float32)

    # remove the label from the feature numpy array
    test_example_batch = np.delete(arr=test_example_batch, obj=[17], axis=1)

    # cast the feature array to float32
    test_example_batch = test_example_batch.astype(np.float32)

    # create initial RNN state array, filled with zeros
    initial_state = np.zeros([BATCH_SIZE, CELL_SIZE])

    # cast the array to float32
    initial_state = initial_state.astype(np.float32)

    # variables initializer
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        
        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
        
        if checkpoint and checkpoint.model_checkpoint_path:
            saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
            print('Loaded model from {}'.format(tf.train.latest_checkpoint(checkpoint_path)))

        try:
            for _ in range(1):
                # todo Get test examples and test labels by batch

                # one-hot encode features according to NUM_BIN
                example_onehot = tf.one_hot(test_example_batch[0:256], NUM_BIN, 1.0, 0.0)
                x_onehot = sess.run(example_onehot)

                # one-hot encode labels according to NUM_CLASSES
                label_onehot = tf.one_hot(test_label_batch[0:256], NUM_CLASSES, 1.0, 0.0)
                y_onehot = sess.run(label_onehot)

                # dictionary for input values for the tensors
                feed_dict = {'input/x_input:0': x_onehot,
                             'initial_state:0': initial_state.astype(np.float32),
                             'p_keep:0': DROPOUT_P_KEEP}

                # get the tensor for classification
                softmax_tensor = sess.graph.get_tensor_by_name('accuracy/Softmax:0')
                predictions = sess.run(softmax_tensor, feed_dict=feed_dict)

                # add key, value pair for labels
                feed_dict['input/y_input:0'] = y_onehot

                # get the tensor for calculating the classification accuracy
                accuracy_tensor = sess.graph.get_tensor_by_name('accuracy/accuracy/Mean:0')
                accuracy = sess.run(accuracy_tensor, feed_dict=feed_dict)

                # concatenate the actual labels to the predicted labels
                prediction_and_actual = np.concatenate((predictions, y_onehot), axis=1)

                # print the full array, may be set to np.nan
                np.set_printoptions(threshold=np.inf)
                print(prediction_and_actual)
                print('Accuracy : {}'.format(accuracy))

                # save the full array
                np.savetxt(result_filename, X=prediction_and_actual, fmt='%.8f', delimiter=',', newline='\n')
        except tf.errors.OutOfRangeError:
            print('EOF')
        except KeyboardInterrupt:
            print('KeyboardInterrupt')


def parse_args():
    parser = argparse.ArgumentParser(description='GRU+Softmax Classifier')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-d', '--dataset', required=True, type=str,
                       help='path of the dataset to be classified')
    group.add_argument('-m', '--model', required=True, type=str,
                       help='path of the trained model')
    group.add_argument('-r', '--result', required=True, type=str,
                       help='filename for the saved result')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()

    predict(args.dataset, args.model, args.result)
