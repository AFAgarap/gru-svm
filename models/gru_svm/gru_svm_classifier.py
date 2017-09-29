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

"""Classifier program based on the GRU+SVM model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.3.8'
__author__ = 'Abien Fred Agarap'

import argparse
from utils.data import load_data
from utils.data import plot_accuracy
import numpy as np
import tensorflow as tf

BATCH_SIZE = 256
CELL_SIZE = 256
DROPOUT_P_KEEP = 1.0
NUM_BIN = 10
NUM_CLASSES = 2


def predict(test_data, checkpoint_path, result_filename):
    """Classifies the data whether there is an attack or none"""

    test_features, test_labels = load_data(dataset=test_data)

    test_size = test_features.shape[0]

    # create initial RNN state array, filled with zeros
    initial_state = np.zeros([BATCH_SIZE, CELL_SIZE])

    # cast the array to float32
    initial_state = initial_state.astype(np.float32)

    # variables initializer
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        # get the checkpoint file
        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

        if checkpoint and checkpoint.model_checkpoint_path:
            # if checkpoint file exists, load the saved meta graph
            saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
            # and restore previously saved variables
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
            print('Loaded model from {}'.format(tf.train.latest_checkpoint(checkpoint_path)))

        accuracy_records = []

        try:
            for step in range(test_size // BATCH_SIZE):

                offset = (step * BATCH_SIZE) % test_size
                test_features_batch = test_features[offset:(offset + BATCH_SIZE)]
                test_labels_batch = test_labels[offset:(offset + BATCH_SIZE)]

                # one-hot encode labels according to NUM_CLASSES
                label_onehot = tf.one_hot(test_labels_batch, NUM_CLASSES, 1.0, -1.0)
                y_onehot = sess.run(label_onehot)

                # dictionary for input values for the tensors
                feed_dict = {'input/x_input:0': test_features_batch,
                             'initial_state:0': initial_state, 'p_keep:0': DROPOUT_P_KEEP}

                # get the tensor for classification
                prediction_tensor = sess.graph.get_tensor_by_name('accuracy/prediction:0')
                predictions = sess.run(prediction_tensor, feed_dict=feed_dict)

                # add key, value pair for labels
                feed_dict['input/y_input:0'] = test_labels_batch

                # get the tensor for calculating the classification accuracy
                accuracy_tensor = sess.graph.get_tensor_by_name('accuracy/accuracy/Mean:0')
                accuracy = sess.run(accuracy_tensor, feed_dict=feed_dict)

                # concatenate the actual labels to the predicted labels
                prediction_and_actual = np.concatenate((predictions, y_onehot), axis=1)

                # print the full array, may be set to np.nan
                np.set_printoptions(threshold=np.inf)
                # print(prediction_and_actual)
                print('step [{}] Accuracy : {}'.format(step, accuracy))

                accuracy_records.append([step, accuracy])

                # save the full array
                # np.savetxt(result_filename, X=prediction_and_actual, fmt='%.1f', delimiter=',', newline='\n')
        except KeyboardInterrupt:
            print('KeyboardInterrupt at step {}'.format(step))
        finally:
            print('Done classifying at step {}'.format(step))

    accuracy_records = np.array(accuracy_records)
    print('Average test accuracy : {}'.format(np.mean(accuracy_records[:, 1])))
    plot_accuracy(accuracy_records)


def parse_args():
    parser = argparse.ArgumentParser(description='GRU+SVM Classifier')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-d', '--test_data', required=True, type=str,
                       help='the NumPy array test data (*.npy) to be classified')
    group.add_argument('-m', '--model', required=True, type=str,
                       help='path of the trained model')
    group.add_argument('-r', '--result', required=True, type=str,
                       help='filename for the saved result')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()

    predict(args.test_data, args.model, args.result)
