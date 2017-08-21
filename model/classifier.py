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

import data
import numpy as np
import tensorflow as tf

BATCH_SIZE = 256
CELL_SIZE = 256
DROPOUT_P_KEEP = 1.0
N_CLASSES = 2
SEQUENCE_LENGTH = 21

CHECKPOINT_PATH = '/home/darth/GitHub Projects/gru_svm/model/checkpoint/gru+svm/'
MODEL_NAME = 'gru_svm.ckpt'

TEST_PATH = '/home/darth/GitHub Projects/gru_svm/dataset/test'


def predict():
    # test_example, test_label = data.input_pipeline(path=TEST_PATH, batch_size=BATCH_SIZE,
    #                                                num_classes=N_CLASSES, num_epochs=1)
    test_file = TEST_PATH + '/24.csv'
    test_example_batch = np.genfromtxt(test_file, delimiter=',')
    test_example_batch = np.delete(arr=test_example_batch, obj=[17], axis=1)

    x_input = tf.placeholder(dtype=tf.float32, shape=[None, SEQUENCE_LENGTH, 10], name='x_input')
    initial_state = np.zeros([BATCH_SIZE, CELL_SIZE])

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        
        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_PATH)
        
        if checkpoint and checkpoint.model_checkpoint_path:
            saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
            saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_PATH))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            for _ in range(1):
                # test_example_batch, test_label_batch = sess.run([test_example, test_label])
                example_onehot = tf.one_hot(test_example_batch[:256].astype(np.float32), 10, 1.0, 0.0)
                x_onehot = sess.run(example_onehot)
                print(x_onehot.shape)
                feed_dict = {x_input: x_onehot,
                             'initial_state:0': initial_state, 'p_keep:0': 1}
                prediction = tf.get_default_graph().get_operation_by_name('accuracy/prediction')
                # prediction_ = sess.run('prediction:1', feed_dict=feed_dict)
                prediction_ = sess.run(prediction, feed_dict=feed_dict)
                print('Prediction: {}'.format(prediction_))
        except tf.errors.OutOfRangeError:
            print('EOF')
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        finally:
            coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    predict()
