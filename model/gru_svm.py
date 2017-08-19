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

"""Implementation of proposed GRU+SVM model for Intrusion Detection"""

__version__ = '0.1.1'
__author__ = 'Abien Fred Agarap'

import data
import math
import numpy as np
import os
import tensorflow as tf
import time

# hyper-parameters
BATCH_SIZE = 512
CELL_SIZE = 512
DROPOUT_P_KEEP = 0.8
HM_EPOCHS = 2
LEARNING_RATE = 0.001
N_CLASSES = 2
P_KEEP = 0.8
SEQUENCE_LENGTH = 21
SVM_C = 0.5

GPU_OPTIONS = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

# tf.train.Saver() parameters
CHECKPOINT_PATH = 'checkpoint/'
MODEL_NAME = 'gru_svm'

LOGS_PATH = 'logs/test/'

TRAIN_PATH = '/home/darth/GitHub Projects/gru_svm/dataset/train'


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


def main():
    examples, labels = data.input_pipeline(path=TRAIN_PATH,
                                           batch_size=BATCH_SIZE,
                                           num_classes=N_CLASSES,
                                           num_epochs=HM_EPOCHS)

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    p_keep = tf.placeholder(tf.float32, name='p_keep')

    with tf.name_scope('input') as input:
        # [BATCH_SIZE, SEQUENCE_LENGTH]
        x_input = tf.placeholder(dtype=tf.float32, shape=[None, SEQUENCE_LENGTH, 10], name='x_input')

        # [BATCH_SIZE, SEQUENCE_LENGTH]
        y_input = tf.placeholder(dtype=tf.float32, shape=[None, N_CLASSES], name='y_input')

    # [BATCH_SIZE, CELL_SIZE]
    state = tf.placeholder(dtype=tf.float32, shape=[None, CELL_SIZE], name='initial_state')

    cell = tf.contrib.rnn.GRUCell(CELL_SIZE)
    drop_cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=p_keep)

    # outputs: [BATCH_SIZE, SEQUENCE_LENGTH, CELL_SIZE]
    # states: [BATCH_SIZE, CELL_SIZE]
    outputs, states = tf.nn.dynamic_rnn(drop_cell, x_input, initial_state=state, dtype=tf.float32)

    states = tf.identity(states, name='H')

    with tf.name_scope('weights'):
        weight = tf.Variable(tf.random_normal([CELL_SIZE, N_CLASSES], stddev=0.01), name='weight')
        variable_summaries(weight)
    with tf.name_scope('biases'):
        bias = tf.Variable(tf.constant(0.1, shape=[N_CLASSES]), name='biases')
        variable_summaries(bias)

    hf = tf.transpose(outputs, [1, 0, 2])
    last = tf.gather(hf, int(hf.get_shape()[0]) - 1)

    with tf.name_scope('Wx_plus_b'):
        output = tf.matmul(last, weight) + bias
        tf.summary.histogram('pre-activations', output)

    with tf.name_scope('loss'):
        regularization_loss = 0.5 * tf.reduce_sum(tf.square(weight))
        hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([BATCH_SIZE, N_CLASSES]), 1 - y_input * output))
        with tf.name_scope('loss'):
            cost = regularization_loss + SVM_C * hinge_loss
    tf.summary.scalar('loss', cost)

    # with tf.name_scope('loss'):
    #     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_input))
    # tf.summary.scalar('loss', cost)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.name_scope('accuracy'):
        predicted_class = tf.sign(output)
        with tf.name_scope('correct_prediction'):
            correct = tf.equal(tf.argmax(predicted_class, 1), tf.argmax(y_input, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    tf.summary.scalar('accuracy', accuracy)

    # accuracy for softmax
    # with tf.name_scope('accuracy'):
    #     with tf.name_scope('correct_prediction'):
    #         correct = tf.equal(tf.argmax(output, 1), tf.argmax(y_input, 1))
    #     with tf.name_scope('accuracy'):
    #         accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    # tf.summary.scalar('accuracy', accuracy)

    if not os.path.exists(CHECKPOINT_PATH):
        os.mkdir(CHECKPOINT_PATH)
    saver = tf.train.Saver(max_to_keep=1000)

    # initialize H (current_state) with values of zeros
    current_state = np.zeros([BATCH_SIZE, CELL_SIZE])

    # variable initializer
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # merge all the summaries collected from TF graph
    merged = tf.summary.merge_all()

    # get the time in seconds since the Epoch
    timestamp = str(math.trunc(time.time()))

    # create an event file to contain the TF graph summaries
    writer = tf.summary.FileWriter(LOGS_PATH + timestamp + '-training', graph=tf.get_default_graph())

    with tf.Session() as sess:

        sess.run(init_op)

        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_PATH)

        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                example_batch, label_batch = sess.run([examples, labels])

                feed_dict = {x_input: example_batch, y_input: label_batch, state: current_state,
                             learning_rate: LEARNING_RATE, p_keep: DROPOUT_P_KEEP}

                summary, _, epoch_loss, next_state = sess.run([merged, optimizer, cost, states], feed_dict=feed_dict)

                accuracy_ = sess.run(accuracy, feed_dict=feed_dict)

                current_state = next_state

                if step % 100 == 0:
                    print('step [{}] loss : {}, accuracy : {}'.format(step, epoch_loss, accuracy_))
                    writer.add_summary(summary, step)
                    saver.save(sess, CHECKPOINT_PATH + MODEL_NAME, global_step=step)

                step += 1
        except tf.errors.OutOfRangeError:
            print('EOF -- training done at step {}'.format(step))
        finally:
            writer.close()
            coord.request_stop()

        coord.join(threads)

        saver = tf.train.Saver()
        saver.save(sess, CHECKPOINT_PATH + MODEL_NAME, global_step=step)

        # test_data, test_labels = data.load_test_data_and_labels(path=TEST_PATH)

        # print('Accuracy : {}'.format(sess.run(
        #     accuracy,
        #     feed_dict={x_onehot: test_data[200], y_input: test_labels[200], state: current_state})))


if __name__ == '__main__':
    main()
