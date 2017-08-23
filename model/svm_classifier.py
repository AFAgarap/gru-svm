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

__version__ = '0.1'
__author__ = 'Abien Fred Agarap'

import numpy as np
import tensorflow as tf

BATCH_SIZE = 256
CELL_SIZE = 256
DROPOUT_P_KEEP = 1.0

CHECKPOINT_PATH = '/home/darth/GitHub Projects/gru_svm/model/checkpoint/gru+svm/'

TEST_PATH = '/home/darth/GitHub Projects/gru_svm/dataset/test'


def predict():
    """Classifies the data whether there is an attack or none"""
    test_file = TEST_PATH + '/24.csv'
    test_example_batch = np.genfromtxt(test_file, delimiter=',')
    test_example_batch = np.delete(arr=test_example_batch, obj=[17], axis=1)

    initial_state = np.zeros([BATCH_SIZE, CELL_SIZE])

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_PATH)

        if checkpoint and checkpoint.model_checkpoint_path:
            saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
            saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_PATH))
            print('Loaded model from {}'.format(tf.train.latest_checkpoint(CHECKPOINT_PATH)))

        try:
            for _ in range(1):
                # todo Get test examples and test labels by batch
                example_onehot = tf.one_hot(test_example_batch[2000:2256].astype(np.float32), 10, 1.0, 0.0)
                x_onehot = sess.run(example_onehot)

                feed_dict = {'input/x_input:0': x_onehot,
                             'initial_state:0': initial_state.astype(np.float32),
                             'p_keep:0': DROPOUT_P_KEEP}

                prediction_tensor = sess.graph.get_tensor_by_name('accuracy/prediction:0')
                predictions = sess.run(prediction_tensor, feed_dict=feed_dict)
                print('Predictions : {}'.format(predictions))
                # todo Get the prediction accuracy
        except tf.errors.OutOfRangeError:
            print('EOF')
        except KeyboardInterrupt:
            print('KeyboardInterrupt')


if __name__ == '__main__':
    predict()
