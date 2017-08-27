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

"""Classifier program based on the GRU+SVM model"""

__version__ = '0.3.2'
__author__ = 'Abien Fred Agarap'

import argparse
import numpy as np
import tensorflow as tf

BATCH_SIZE = 256
CELL_SIZE = 256
DROPOUT_P_KEEP = 1.0
NUM_BIN = 10
NUM_CLASSES = 2


def predict(test_path, checkpoint_path):
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

        # get the checkpoint file
        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

        if checkpoint and checkpoint.model_checkpoint_path:
            # if checkpoint file exists, load the saved meta graph
            saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
            # and restore previously saved variables
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
            print('Loaded model from {}'.format(tf.train.latest_checkpoint(checkpoint_path)))

        try:
            for _ in range(1):
                # todo Get test examples and test labels by batch

                # one-hot encode features according to NUM_BIN
                example_onehot = tf.one_hot(test_example_batch[2000:2256], NUM_BIN, 1.0, 0.0)
                x_onehot = sess.run(example_onehot)

                # one-hot encode labels according to NUM_CLASSES
                label_onehot = tf.one_hot(test_label_batch[2000:2256], NUM_CLASSES, 1.0, -1.0)
                y_onehot = sess.run(label_onehot)

                # dictionary for input values for the tensors
                feed_dict = {'input/x_input:0': x_onehot,
                             'initial_state:0': initial_state, 'p_keep:0': DROPOUT_P_KEEP}

                # get the tensor for classification
                prediction_tensor = sess.graph.get_tensor_by_name('accuracy/prediction:0')
                predictions = sess.run(prediction_tensor, feed_dict=feed_dict)

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
                np.savetxt('svm_results.csv', X=prediction_and_actual, fmt='%.8f', delimiter=',', newline='\n')
        except tf.errors.OutOfRangeError:
            print('EOF')
        except KeyboardInterrupt:
            print('KeyboardInterrupt')


def parse_args():
    parser = argparse.ArgumentParser(description='GRU+SVM Classifier')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-d', '--dataset', required=True, type=str,
                       help='path of the dataset to be classified')
    group.add_argument('-m', '--model', required=True, type=str,
                       help='path of the trained model')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()

    predict(args.dataset, args.model)
