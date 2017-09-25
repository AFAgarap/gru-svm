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

"""Classifier program based on the SVM model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.3'
__author__ = 'Abien Fred Agarap'

import argparse
from data import load_data
import numpy as np
import os
import tensorflow as tf

BATCH_SIZE = 256
CELL_SIZE = 256
NUM_CLASSES = 2


def predict(test_data, checkpoint_path, result_path):
    """Classifies the data whether there is an attack or none"""

    test_features, test_labels = load_data(dataset=test_data)

    test_size = test_features.shape[0]
    print(test_size)

    # variables initializer
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

        # check if trained model exists
        if checkpoint and checkpoint.model_checkpoint_path:
            # load the trained model
            saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
            # restore the variables
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
            print('Loaded model from {}'.format(tf.train.latest_checkpoint(checkpoint_path)))

        try:
            for step in range(test_size // BATCH_SIZE):
                offset = (step * BATCH_SIZE) % test_size
                test_example_batch = test_features[offset:(offset + BATCH_SIZE)]
                test_label_batch = test_labels[offset:(offset+BATCH_SIZE)]

                # dictionary for input values for the tensors
                feed_dict = {'input/x_input:0': test_example_batch}

                # get the tensor for classification
                svm_tensor = sess.graph.get_tensor_by_name('accuracy/prediction:0')
                predictions = sess.run(svm_tensor, feed_dict=feed_dict)

                label_onehot = tf.one_hot(test_label_batch, NUM_CLASSES, 1.0, -1.0)
                y_onehot = sess.run(label_onehot)

                # add key, value pair for labels
                feed_dict['input/y_input:0'] = test_label_batch

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
                np.savetxt(os.path.join(result_path, 'svm-{}-result.csv'.format(step)), X=prediction_and_actual,
                           fmt='%.1f', delimiter=',', newline='\n')
        except tf.errors.OutOfRangeError:
            print('EOF')
        except KeyboardInterrupt:
            print('KeyboardInterrupt at step {}'.format(step))


def parse_args():
    parser = argparse.ArgumentParser(description='SVM Classifier')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-d', '--dataset', required=True, type=str,
                       help='the NumPy array dataset (*.npy) to be classified')
    group.add_argument('-m', '--model', required=True, type=str,
                       help='path of the trained model')
    group.add_argument('-r', '--result_path', required=True, type=str,
                       help='path where to save the actual and predicted labels')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()

    predict(args.dataset, args.model, args.result_path)
